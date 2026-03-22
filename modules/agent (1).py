# agent.py

import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from typing import Dict, List, Optional, Any, Tuple
from .inventory import PantryManager
from .rag import RecipeRAG
from .database import Database

# Словарь замен для продуктов (как в вашем коде)
INGREDIENT_REPLACEMENTS = {
    "куриное филе": ["индейка", "куриная грудка", "телятина"],
    "сметана": ["йогурт", "кефир", "сливки"],
    "масло оливковое": ["масло растительное"],
    "сахар": ["мед", "стевия"],
    "молоко": ["кефир", "ряженка"],
    "паста": ["макароны", "рис", "гречка"]
}

# АГЕНТ №1: ИНВЕНТАРНЫЙ АГЕНТ

class InventoryAgent:
    """Специализированный агент для управления запасами и поиска альтернатив.
    Работает в режиме "спросили-ответили", но может быть расширен для циклического поиска."""
    
    def __init__(self, pantry_manager: PantryManager):
        self.pantry = pantry_manager

    def smart_check(self, recipe_text: str) -> str:
        """Проверяет продукты из рецепта и интеллектуально подбирает замены.
        Возвращает отчёт о наличии и необходимых покупках."""
        
        needed_ingredients = self.pantry.parse_free_text(recipe_text)
        
        # Если не распарсилось, пробуем разбить по запятым
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

# АГЕНТ №2: НУТРИЦИОННЫЙ АГЕНТ (ГЛАВНЫЙ)
# Реализует полноценный ReAct-цикл для планирования питания

class NutritionAgent:
    """Главный агент-диетолог, реализующий ReAct-цикл для составления рациона.
    На каждом шаге: мысль → действие → наблюдение → повтор, пока не будет готов ответ.
    Может вызывать инвентарного агента для проверки наличия продуктов."""
    
    def __init__(self, db: Database, rag: RecipeRAG, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.db = db
        self.rag = rag
        self.current_user_id = None
        self.inventory_agent = None
        
        # Инициализация LLM
        self.tokenizer = None
        self.model = None
        self._load_llm(model_id)
        
        # Максимальное количество итераций в цикле
        self.max_iterations = 5

    def _load_llm(self, model_id: str):
        """Загружает модель с 4-битным квантованием"""
        print(f"Загрузка модели {model_id}...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        print("✅ Модель загружена!")

    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024) -> str:
        """Генерирует ответ модели на основе истории сообщений"""
        # Применяем чат-шаблон
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Токенизируем
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Генерируем
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        
        # Декодируем только новую часть
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

    def _get_user_profile_text(self, user_id: int) -> str:
        """Возвращает текстовое описание профиля пользователя"""
        user = self.db.get_user(user_id)
        if not user:
            return ""
        
        return f"""
        🎯 Цель: {user['goal']}
        🥗 Тип питания: {user['diet']}
        🚫 Аллергии: {user['allergies'] if user['allergies'] else 'нет'}
        📊 Антропометрия: {user['age']} лет, {user['weight']} кг, {user['height']} см, пол: {user['gender']}
        🏃 Активность: {user['activity']}
        🔥 Норма калорий: {user['calories_target']} ккал
        🍗 БЖУ: {user['protein_target']}г / {user['fat_target']}г / {user['carbs_target']}г
        """

    def _create_system_prompt(self, user_id: int) -> str:
        """Создаёт системный промпт для ReAct-цикла"""
        profile_text = self._get_user_profile_text(user_id)
        
        return f"""Ты — ИИ-диетолог. Твоя задача — составлять сбалансированные планы питания на день.

У тебя есть доступ к следующим инструментам:

1. **search_recipes(query)** — ищет рецепты по запросу. Запрос может описывать желаемые блюда, ингредиенты, тип питания.
2. **check_inventory(recipe_text)** — проверяет наличие продуктов для рецепта и предлагает замены. Передай этому инструменту список продуктов из рецепта.

ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:
{profile_text}

Для выполнения задачи используй следующий формат:

Мысль: [твои рассуждения о том, что нужно сделать]
Действие: [название инструмента: search_recipes или check_inventory]
Аргумент: [аргумент для инструмента]
Наблюдение: [результат выполнения инструмента — я предоставлю его автоматически]
... (повторяй цикл Мысль → Действие → Наблюдение, пока не соберёшь всю информацию)
Ответ: [финальный ответ пользователю с полным планом питания]

ПРАВИЛА:
1. Сначала используй search_recipes, чтобы найти подходящие рецепты.
2. Затем используй check_inventory для проверки наличия продуктов.
3. Если продуктов не хватает, учти это в ответе или предложи альтернативы.
4. Финальный ответ должен содержать ЗАВТРАК, ОБЕД и УЖИН с названиями блюд, ингредиентами и краткими инструкциями.
5. В конце ответа добавь комментарий о соответствии КБЖУ дневным нормам.

ВАЖНО: Не выдумывай наблюдения — они будут подставлены автоматически после выполнения действий.
"""

    def search_recipes(self, query: str) -> str:
        """Поиск рецептов через RAG"""
        user = self.db.get_user(self.current_user_id)
        profile = None
        if user:
            profile = {
                'allergies': user['allergies'],
                'diet': user['diet']
            }
        meal_plan = self.rag.build_meal_plan(query, profile)
        return self.rag.format_meal_plan(meal_plan)

    def check_inventory(self, recipe_text: str) -> str:
        """Проверка наличия продуктов через инвентарного агента"""
        if self.inventory_agent:
            return self.inventory_agent.smart_check(recipe_text)
        return "Инвентарный агент недоступен"

    def _extract_action(self, text: str) -> Optional[Tuple[str, str]]:
        """Извлекает действие и аргумент из текста"""
        # Ищем паттерн "Действие: xxx" и "Аргумент: yyy"
        action_match = re.search(r'Действие:\s*(\w+)', text, re.IGNORECASE)
        arg_match = re.search(r'Аргумент:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
        
        if action_match and arg_match:
            action = action_match.group(1).strip()
            argument = arg_match.group(1).strip()
            return action, argument
        return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Извлекает финальный ответ из текста"""
        # Ищем "Ответ:" в конце
        answer_match = re.search(r'Ответ:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Если нет явного "Ответ:", но текст выглядит как ответ
        if 'завтрак' in text.lower() and 'обед' in text.lower() and 'ужин' in text.lower():
            return text.strip()
        
        return None

    def run(self, user_id: int, user_query: str) -> str:
        """Запускает ReAct-цикл агента для составления плана питания"""
        
        self.current_user_id = user_id
        
        # Инициализируем инвентарного агента
        pantry = PantryManager(self.db, user_id)
        self.inventory_agent = InventoryAgent(pantry)
        
        # Строим начальные сообщения
        system_prompt = self._create_system_prompt(user_id)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос: {user_query}"}
        ]
        
        # Добавляем историю диалога (последние 2 сообщения)
        history = self.db.get_chat_history(user_id, limit=2)
        for role, content in history:
            messages.append({"role": role, "content": content})
        
        print(f"\n🏁 СТАРТ ReAct-цикла")
        print(f"📝 Запрос: {user_query}\n")
        print("-" * 60)
        
        # Основной ReAct-цикл
        iteration = 0
        full_response = ""
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 Итерация {iteration}/{self.max_iterations}")
            
            # Генерируем ответ модели
            response = self._generate(messages)
            full_response = response
            
            print(f"🤖 Модель:\n{response}\n")
            
            # Проверяем, есть ли финальный ответ
            final_answer = self._extract_final_answer(response)
            if final_answer:
                print("✅ Найден финальный ответ, завершаем цикл")
                # Сохраняем в историю
                self.db.add_chat_message(user_id, "user", user_query)
                self.db.add_chat_message(user_id, "assistant", final_answer)
                return final_answer
            
            # Извлекаем действие
            action_info = self._extract_action(response)
            
            if not action_info:
                print("⚠️ Не удалось извлечь действие, просим модель уточнить")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user", 
                    "content": "Пожалуйста, укажи действие в формате 'Действие: ...' и 'Аргумент: ...'"
                })
                continue
            
            action, argument = action_info
            print(f"⚙️ Выполнение: {action}({argument})")
            
            # Выполняем действие
            if action == "search_recipes":
                observation = self.search_recipes(argument)
            elif action == "check_inventory":
                observation = self.check_inventory(argument)
            else:
                observation = f"Неизвестное действие: {action}. Доступны: search_recipes, check_inventory"
            
            print(f"👁️ Наблюдение: {observation[:500]}...")
            print("-" * 60)
            
            # Добавляем результат в историю
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Наблюдение: {observation}"})
        
        # Если цикл завершился без финального ответа
        print("⚠️ Достигнуто максимальное количество итераций")
        return full_response if full_response else "Не удалось составить план питания. Попробуйте уточнить запрос."

# АГЕНТ №3: МАСТЕР-АГЕНТ (ОРКЕСТРАТОР)
# Для реализации мультиагентности: управляет специализированными агентами

class MasterAgent:
    """Мастер-агент, который управляет специализированными агентами.
    Реализует паттерн "оркестратор" из лекции — распределяет задачи между агентами."""
    
    def __init__(self, db: Database, rag: RecipeRAG, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.db = db
        self.rag = rag
        self.model_id = model_id
        
        # Подчинённые агенты (будут созданы при необходимости)
        self.nutrition_agent = None
        self.inventory_agent = None
        
        # Инициализация LLM для оркестрации
        self._init_llm()
        
        self.max_iterations = 3

    def _init_llm(self):
        """Инициализирует LLM для оркестрации"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """Генерирует ответ для оркестрации"""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def run(self, user_id: int, user_query: str) -> str:
        """Мастер-агент определяет, какой специализированный агент нужен, и делегирует задачу"""
        
        print(f"\n🎭 МАСТЕР-АГЕНТ")
        print(f"📝 Запрос: {user_query}\n")
        
        # Создаём системный промпт для оркестрации
        system_prompt = """
        Ты — мастер-агент, который управляет специализированными помощниками.
        
        Доступные специализированные агенты:
        1. **nutrition_agent** — специалист по составлению планов питания. Используй его для задач, связанных с:
           - составлением меню на день/неделю
           - подбором рецептов под диетические ограничения
           - расчётом КБЖУ
        
        2. **inventory_agent** — специалист по управлению запасами. Используй его для задач, связанных с:
           - проверкой наличия продуктов
           - подбором замен для отсутствующих продуктов
           - формированием списка покупок
        
        Формат ответа:
        - Если задача требует одного специалиста: просто верни имя агента и запрос.
        - Если задача сложная и требует нескольких специалистов: верни план последовательного вызова.
        
        Отвечай строго в формате:
        Агент: [имя_агента]
        Запрос: [текст запроса для агента]
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Добавляем историю
        history = self.db.get_chat_history(user_id, limit=2)
        for role, content in history:
            messages.append({"role": role, "content": content})
        
        # Получаем решение мастера
        response = self._generate(messages)
        print(f"🎯 Решение мастера:\n{response}\n")
        
        # Парсим ответ мастера
        agent_match = re.search(r'Агент:\s*(\w+)', response, re.IGNORECASE)
        query_match = re.search(r'Запрос:\s*(.+?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
        
        if not agent_match:
            # Если мастер не смог определить агента, используем нутриционного
            print("⚠️ Мастер не определил агента, используем NutritionAgent")
            if not self.nutrition_agent:
                self.nutrition_agent = NutritionAgent(self.db, self.rag, self.model_id)
            return self.nutrition_agent.run(user_id, user_query)
        
        agent_name = agent_match.group(1).strip()
        agent_query = query_match.group(1).strip() if query_match else user_query
        
        # Делегируем задачу соответствующему агенту
        if agent_name.lower() == "inventory_agent":
            print(f"📦 Делегируем InventoryAgent: {agent_query}")
            pantry = PantryManager(self.db, user_id)
            if not self.inventory_agent:
                self.inventory_agent = InventoryAgent(pantry)
            return self.inventory_agent.smart_check(agent_query)
        
        else:  # nutrition_agent или по умолчанию
            print(f"🍽️ Делегируем NutritionAgent: {agent_query}")
            if not self.nutrition_agent:
                self.nutrition_agent = NutritionAgent(self.db, self.rag, self.model_id)
            return self.nutrition_agent.run(user_id, agent_query)