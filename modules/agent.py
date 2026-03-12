# agent.py

import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.tools import Tool
from .inventory import PantryManager
from .rag import RecipeRAG
from .database import Database

class NutritionAgent:
    def __init__(self, db: Database, rag: RecipeRAG, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.db = db
        self.rag = rag
        self.llm = self._load_llm(model_id)
        self.current_user_id = None
        self.current_pantry = None
        self.tools = self._create_tools()

    def _load_llm(self, model_id):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0.1, do_sample=False)
        return HuggingFacePipeline(pipeline=pipe)

    def _create_tools(self):
        def inventory_check(query: str) -> str:
            if self.current_pantry is None:
                return "Ошибка: пользователь не определён."
            available = self.current_pantry.get_available()
            if not available:
                return "В холодильнике пока пусто."
            return "В наличии: " + ", ".join([f"{name}: {qty:.1f} (в базовых ед.)" for name, qty in available.items()])

        def recipe_search(query: str) -> str:
            user = self.db.get_user(self.current_user_id)
            profile = None
            if user:
                profile = {
                    'allergies': user['allergies'],
                    'diet': user['diet']
                }
            meal_plan = self.rag.build_meal_plan(query, profile)
            return self.rag.format_meal_plan(meal_plan)

        return {
            "Inventory_Check": inventory_check,
            "Recipe_Search": recipe_search
        }

    def run(self, user_id: int, user_query: str) -> str:
        self.current_user_id = user_id
        self.current_pantry = PantryManager(self.db, user_id)

        user = self.db.get_user(user_id)
        nutrition_info = ""
        if user:
            nutrition_info = (
                f"Дневные нормы пользователя:\n"
                f"- Калории: {user['calories_target']} ккал\n"
                f"- Белки: {user['protein_target']} г\n"
                f"- Жиры: {user['fat_target']} г\n"
                f"- Углеводы: {user['carbs_target']} г\n\n"
            )

        # Берём только последние 2 сообщения из истории
        history = self.db.get_chat_history(user_id, limit=2)
        history_text = "\n".join([f"{h[0]}: {h[1]}" for h in history])

        # Жёстко обрезаем историю, если она слишком длинная
        MAX_HISTORY_CHARS = 1000
        if len(history_text) > MAX_HISTORY_CHARS:
            history_text = history_text[-MAX_HISTORY_CHARS:] + "\n... (история обрезана)"

        system_prompt = f"""Ты — ИИ-диетолог. У тебя есть доступ к базе рецептов (Recipe_Search) и твоему холодильнику (Inventory_Check).
Твоя задача: сначала проверить, что есть дома, а потом предложить подходящий рецепт или рацион.

{nutrition_info}

Формат ответа должен строго содержать три приёма пищи: ЗАВТРАК, ОБЕД, УЖИН. 
Если какое-то блюдо не найдено, укажи это явно (например, "Ужин: не найдено").

Для каждого блюда укажи:
- Название
- Ингредиенты
- Краткое приготовление

В конце ответа обязательно укажи дневные нормы пользователя (калории, белки, жиры, углеводы) и кратко прокомментируй, соответствует ли им предложенный рацион.

Формат взаимодействия:
Мысль: [твои рассуждения]
Действие: [название инструмента]
Аргумент: [запрос]
Ответ: [финальный текст на русском]

ВАЖНО: Если ты уже дал ответ, больше не вызывай инструменты. Завершай диалог.

История:
{history_text}
"""
        full_prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_query}\n<|im_end|>\n<|im_start|>assistant\nМысль:"

        response = self.llm.invoke(full_prompt)
        final_answer = self._execute_agent_cycle(full_prompt, response)
        self.db.add_chat_message(user_id, "user", user_query)
        self.db.add_chat_message(user_id, "assistant", final_answer)
        return final_answer

    def _execute_agent_cycle(self, initial_prompt: str, first_response: str) -> str:
        full_text = initial_prompt + first_response
        max_iterations = 2
        last_response = ""

        for i in range(max_iterations):
            # Если в тексте уже есть "Ответ:", пытаемся извлечь
            if "Ответ:" in full_text:
                # Ищем последний "Ответ:"
                parts = full_text.split("Ответ:")
                candidate = parts[-1].strip()
                # Если ответ не пустой и содержит три приёма пищи (хотя бы упоминания)
                if candidate and any(word in candidate.lower() for word in ['завтрак', 'обед', 'ужин']):
                    return candidate

            action_match = re.search(r"Действие:\s*(.*?)\s*Аргумент:\s*(.*?)\s*(?=\n|$)", full_text, re.DOTALL)
            if not action_match:
                # Если нет действия, но есть ответ
                answer_match = re.search(r"Ответ:\s*(.*?)$", full_text, re.DOTALL)
                if answer_match:
                    candidate = answer_match.group(1).strip()
                    if candidate:
                        return candidate
                else:
                    return first_response.strip()

            action = action_match.group(1).strip()
            argument = action_match.group(2).strip()

            if action in self.tools:
                observation = self.tools[action](argument)
            else:
                observation = f"Неизвестное действие: {action}"

            full_text += f"\nРезультат: {observation}\n"
            full_text += "Мысль:"

            next_part = self.llm.invoke(full_text)
            full_text += next_part

            # Защита от повторений
            if next_part.strip() == last_response:
                break
            last_response = next_part.strip()

        # Если превысили лимит, пытаемся извлечь последний ответ
        answer_match = re.search(r"Ответ:\s*(.*?)$", full_text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return "Не удалось получить ответ. Попробуйте ещё раз."