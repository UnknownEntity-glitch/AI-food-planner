# agent.py

import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.tools import Tool
from .inventory import PantryManager
from .rag import RecipeRAG
from .database import Database

INGREDIENT_REPLACEMENTS = {
    "куриное филе": ["индейка", "куриная грудка", "телятина"],
    "сметана": ["йогурт", "кефир", "сливки"],
    "масло оливковое": ["масло растительное"],
    "сахар": ["мед", "стевия"],
    "молоко": ["кефир", "ряженка"],
    "паста": ["макароны", "рис", "гречка"]
}

# АГЕНТ №1: INVENTORY AGENT
class InventoryAgent:
    """Специализированный агент для управления запасами и поиска альтернатив."""
    def __init__(self, pantry_manager: PantryManager):
        self.pantry = pantry_manager

    def smart_check(self, recipe_text: str) -> str:
        """Проверяет продукты из рецепта и интеллектуально подбирает замены."""
        needed_ingredients = self.pantry.parse_free_text(recipe_text)
        
        if not needed_ingredients:
            parts = [p.strip() for p in recipe_text.split(',')]
            for part in parts:
                ing = self.pantry.parse_free_text(part)
                if ing:
                    needed_ingredients.extend(ing)
        
        available_stock = self.pantry.get_available()
        
        if not available_stock:
            return "В холодильнике пусто. Нужно купить всё по списку."

        shopping_list = []
        from_fridge = []
        notes = []

        for ing in needed_ingredients:
            name = ing['canonical']
            needed_qty = ing['qty_base'] or ing['qty'] or 1.0
            unit = ing['unit_base'] or ing['unit'] or 'шт'
            
            in_stock = available_stock.get(name, 0)
            
            if in_stock >= needed_qty - 1e-6:
                from_fridge.append(f"{name} ({needed_qty:.1f} {unit})")
            else:
                replacement = None
                alternatives = INGREDIENT_REPLACEMENTS.get(name, [])
                for alt in alternatives:
                    if available_stock.get(alt, 0) >= needed_qty - 1e-6:
                        replacement = alt
                        break
                
                if replacement:
                    from_fridge.append(f"{replacement} (вместо {name})")
                    notes.append(f"Заменил {name} на {replacement}")
                else:
                    to_buy_qty = needed_qty - in_stock
                    if to_buy_qty > 0:
                        shopping_list.append(f"{name}: {to_buy_qty:.1f} {unit}")

        result = "📋 Отчет Агента по запасам:\n"
        if from_fridge:
            result += f"✅ Использовать из дома: {', '.join(from_fridge)}\n"
        if shopping_list:
            result += f"🛒 Нужно купить: {', '.join(shopping_list)}\n"
        if notes:
            result += f"💡 Заметки по заменам: {'; '.join(notes)}\n"
        
        return result if result != "📋 Отчет Агента по запасам:\n" else "Все продукты уже есть в наличии."

# АГЕНТ №2: NUTRITION AGENT
class NutritionAgent:
    """Главный агент, отвечающий за рацион и координацию с InventoryAgent."""
    def __init__(self, db: Database, rag: RecipeRAG, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.db = db
        self.rag = rag
        self.llm = self._load_llm(model_id)
        self.current_user_id = None
        self.inventory_specialist = None

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

    def search_recipes(self, query: str) -> str:
        """Вызов RAG для поиска рецептов."""
        user = self.db.get_user(self.current_user_id)
        profile = None
        if user:
            profile = {
                'allergies': user['allergies'],
                'diet': user['diet']
            }
        meal_plan = self.rag.build_meal_plan(query, profile)
        return self.rag.format_meal_plan(meal_plan)

    def _create_tools(self):
        """Регистрация инструментов, включая вызов другого агента."""
        return {
            "Recipe_Search": self.search_recipes,
            "Consult_Inventory_Agent": self.inventory_specialist.smart_check
        }

    def run(self, user_id: int, user_query: str) -> str:
        self.current_user_id = user_id
        
        pantry = PantryManager(self.db, user_id)
        self.inventory_specialist = InventoryAgent(pantry)
        
        self.tools = self._create_tools()

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

        history = self.db.get_chat_history(user_id, limit=2)
        history_text = "\n".join([f"{h[0]}: {h[1]}" for h in history])

        system_prompt = f"""Ты — ИИ-диетолог. Твоя задача: планировать рацион.
У тебя есть доступ к:
1. Recipe_Search — поиск рецептов.
2. Consult_Inventory_Agent — связь с Агентом Запасов (он проверяет продукты дома и предлагает замены).

ВАЖНО: Перед тем как предложить рецепт, обязательно вызывай Consult_Inventory_Agent для проверки наличия продуктов.
Формат вызова:
Действие: Consult_Inventory_Agent
Аргумент: [список продуктов из рецепта]

Пример:
Мысль: Надо проверить, что есть дома для приготовления курицы с рисом.
Действие: Consult_Inventory_Agent
Аргумент: куриное филе 200 г, рис 100 г, соль, перец
Результат: [ответ агента]
Мысль: На основе результатов составляю рацион...
Ответ: ...

{nutrition_info}

Формат ответа: ЗАВТРАК, ОБЕД, УЖИН. Укажи название, ингредиенты и приготовление.
В конце прокомментируй соответствие КБЖУ.

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
        max_iterations = 3
        last_response = ""

        for i in range(max_iterations):
            if "Ответ:" in full_text:
                parts = full_text.split("Ответ:")
                candidate = parts[-1].strip()
                if candidate and any(word in candidate.lower() for word in ['завтрак', 'обед', 'ужин']):
                    return candidate

            action_match = re.search(r"Действие:\s*(.*?)\s*Аргумент:\s*(.*?)\s*(?=\n|$)", full_text, re.DOTALL)
            if not action_match:
                answer_match = re.search(r"Ответ:\s*(.*?)$", full_text, re.DOTALL)
                return answer_match.group(1).strip() if answer_match else first_response.strip()

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

            if next_part.strip() == last_response:
                break
            last_response = next_part.strip()

        answer_match = re.search(r"Ответ:\s*(.*?)$", full_text, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else "Не удалось получить ответ."