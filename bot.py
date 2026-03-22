import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from typing import List
from config import BOT_TOKEN, DATA_DIR, RECIPES_ZIP, USE_AGENT
from modules.database import Database
from modules.inventory import PantryManager
from modules.rag import RecipeRAG
from modules.agent import NutritionAgent, MasterAgent

# В начале файла:
if USE_AGENT:
    if USE_MASTER_AGENT:
        agent = MasterAgent(db, rag)
    else:
        agent = NutritionAgent(db, rag)
else:
    agent = None


bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

db = Database()
rag = RecipeRAG(DATA_DIR, RECIPES_ZIP)
if USE_AGENT:
    agent = NutritionAgent(db, rag)
else:
    agent = None

class ProfileStates(StatesGroup):
    waiting_for_goal = State()
    waiting_for_diet = State()
    waiting_for_allergies = State()
    waiting_for_age = State()
    waiting_for_weight = State()
    waiting_for_height = State()
    waiting_for_gender = State()
    waiting_for_activity = State()
    waiting_for_nutrient_choice = State()  # выбор: авто или ручной

class ManualNutrientStates(StatesGroup):
    waiting_for_calories = State()
    waiting_for_protein = State()
    waiting_for_fat = State()
    waiting_for_carbs = State()

class InventoryStates(StatesGroup):
    waiting_for_items = State()
    confirm_items = State()

class MenuStates(StatesGroup):
    waiting_for_query = State()

def get_main_menu():
    buttons = [
        [KeyboardButton(text="🍎 Составить меню"), KeyboardButton(text="🛒 Список покупок")],
        [KeyboardButton(text="🧊 Мои продукты"), KeyboardButton(text="👤 Мой профиль")]
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)

# Улучшенная функция автоматического расчёта калорий и БЖУ
def calculate_targets(age: int, weight: float, height: float, gender: str, activity: str, goal: str):
    # 1. Базальный метаболизм (Миффлин-Сан Жеор)
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # 2. Коэффициент активности
    activity_mult = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    tdee = bmr * activity_mult.get(activity, 1.2)

    # 3. Корректировка калорий под цель
    if goal == 'Похудение':
        calories = tdee * 0.85
    elif goal == 'Набор массы':
        calories = tdee * 1.1
    else:
        calories = tdee

    # 4. Белок (г/кг веса) с учётом пола и возраста
    if gender == 'male':
        protein_factor = 1.8 if goal == 'Набор массы' else (2.0 if goal == 'Похудение' else 1.5)
    else:
        protein_factor = 1.6 if goal == 'Набор массы' else (1.8 if goal == 'Похудение' else 1.3)

    if age > 50:
        protein_factor *= 0.9

    if activity in ['active', 'very_active']:
        protein_factor += 0.2

    protein_g = round(weight * protein_factor)

    # 5. Жиры (г/кг веса) с учётом пола
    if gender == 'male':
        fat_factor = 1.0 if goal != 'Похудение' else 0.8
    else:
        fat_factor = 1.1 if goal != 'Похудение' else 0.9

    fat_g = round(weight * fat_factor)

    # 6. Углеводы — остаток
    calories_from_protein = protein_g * 4
    calories_from_fat = fat_g * 9
    remaining = calories - (calories_from_protein + calories_from_fat)
    carbs_g = round(max(0, remaining / 4))

    # Защита от отрицательных углеводов
    if carbs_g < 0:
        fat_g = round(weight * 0.8)
        calories_from_fat = fat_g * 9
        remaining = calories - (calories_from_protein + calories_from_fat)
        carbs_g = round(max(0, remaining / 4))

        if carbs_g < 0:
            protein_g = round(weight * 1.2)
            calories_from_protein = protein_g * 4
            remaining = calories - (calories_from_protein + calories_from_fat)
            carbs_g = round(max(0, remaining / 4))

    return round(calories), protein_g, fat_g, carbs_g

# ---------- Хендлеры ----------
@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    user = db.get_user(message.from_user.id)
    if user:
        await message.answer(f"С возвращением, {message.from_user.first_name}!", reply_markup=get_main_menu())
    else:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📉 Похудение", callback_data="goal_Похудение")],
            [InlineKeyboardButton(text="⚖️ Поддержание", callback_data="goal_Поддержание")],
            [InlineKeyboardButton(text="📈 Набор массы", callback_data="goal_Набор массы")]
        ])
        await message.answer("Выберите вашу основную цель:", reply_markup=kb)
        await state.set_state(ProfileStates.waiting_for_goal)

@dp.callback_query(ProfileStates.waiting_for_goal)
async def process_goal(callback: types.CallbackQuery, state: FSMContext):
    goal = callback.data.split("_")[1]
    await state.update_data(goal=goal)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🥩 Обычная", callback_data="diet_Обычная")],
        [InlineKeyboardButton(text="🥦 Вегетарианская", callback_data="diet_Вегетарианская")],
        [InlineKeyboardButton(text="🥑 Кето", callback_data="diet_Кето")]
    ])
    await callback.message.edit_text("Ваш тип питания:", reply_markup=kb)
    await state.set_state(ProfileStates.waiting_for_diet)

@dp.callback_query(ProfileStates.waiting_for_diet)
async def process_diet(callback: types.CallbackQuery, state: FSMContext):
    diet = callback.data.split("_")[1]
    await state.update_data(diet=diet)
    await callback.message.edit_text("Есть ли у вас аллергии? Напишите их текстом (или 'Нет'):")
    await state.set_state(ProfileStates.waiting_for_allergies)

@dp.message(ProfileStates.waiting_for_allergies)
async def process_allergies(message: types.Message, state: FSMContext):
    allergies = message.text
    await state.update_data(allergies=allergies)
    await message.answer("Сколько вам лет?")
    await state.set_state(ProfileStates.waiting_for_age)

@dp.message(ProfileStates.waiting_for_age)
async def process_age(message: types.Message, state: FSMContext):
    try:
        age = int(message.text)
        if age < 10 or age > 120:
            raise ValueError
    except:
        await message.answer("Пожалуйста, введите корректный возраст (число от 10 до 120).")
        return
    await state.update_data(age=age)
    await message.answer("Ваш вес в кг (например, 70.5):")
    await state.set_state(ProfileStates.waiting_for_weight)

@dp.message(ProfileStates.waiting_for_weight)
async def process_weight(message: types.Message, state: FSMContext):
    try:
        weight = float(message.text.replace(',', '.'))
        if weight < 20 or weight > 300:
            raise ValueError
    except:
        await message.answer("Пожалуйста, введите корректный вес (число от 20 до 300).")
        return
    await state.update_data(weight=weight)
    await message.answer("Ваш рост в см (например, 175):")
    await state.set_state(ProfileStates.waiting_for_height)

@dp.message(ProfileStates.waiting_for_height)
async def process_height(message: types.Message, state: FSMContext):
    try:
        height = float(message.text)
        if height < 100 or height > 250:
            raise ValueError
    except:
        await message.answer("Пожалуйста, введите корректный рост (число от 100 до 250).")
        return
    await state.update_data(height=height)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👨 Мужской", callback_data="gender_male")],
        [InlineKeyboardButton(text="👩 Женский", callback_data="gender_female")]
    ])
    await message.answer("Ваш пол:", reply_markup=kb)
    await state.set_state(ProfileStates.waiting_for_gender)

@dp.callback_query(ProfileStates.waiting_for_gender)
async def process_gender(callback: types.CallbackQuery, state: FSMContext):
    gender = callback.data.split("_")[1]
    await state.update_data(gender=gender)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🪑 Сидячий", callback_data="activity_sedentary")],
        [InlineKeyboardButton(text="🚶 Лёгкая активность", callback_data="activity_light")],
        [InlineKeyboardButton(text="🏃 Средняя активность", callback_data="activity_moderate")],
        [InlineKeyboardButton(text="💪 Высокая активность", callback_data="activity_active")],
        [InlineKeyboardButton(text="🏋️ Очень высокая", callback_data="activity_very_active")]
    ])
    await callback.message.edit_text("Уровень физической активности:", reply_markup=kb)
    await state.set_state(ProfileStates.waiting_for_activity)

@dp.callback_query(ProfileStates.waiting_for_activity)
async def process_activity(callback: types.CallbackQuery, state: FSMContext):
    activity = callback.data.split("_")[1]
    await state.update_data(activity=activity)
    # Предлагаем выбор: автоматический расчёт или ручной ввод
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Автоматический расчёт", callback_data="nutrient_auto")],
        [InlineKeyboardButton(text="✍️ Ввести вручную", callback_data="nutrient_manual")]
    ])
    await callback.message.edit_text("Как вы хотите определить дневные нормы калорий и БЖУ?", reply_markup=kb)
    await state.set_state(ProfileStates.waiting_for_nutrient_choice)

@dp.callback_query(ProfileStates.waiting_for_nutrient_choice, F.data == "nutrient_auto")
async def process_nutrient_auto(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    calories, protein, fat, carbs = calculate_targets(
        age=data['age'],
        weight=data['weight'],
        height=data['height'],
        gender=data['gender'],
        activity=data['activity'],
        goal=data['goal']
    )
    # Сохраняем пользователя в БД
    db.create_user(
        telegram_id=callback.from_user.id,
        goal=data['goal'],
        diet=data['diet'],
        allergies=data['allergies'],
        age=data['age'],
        weight=data['weight'],
        height=data['height'],
        gender=data['gender'],
        activity=data['activity'],
        calories_target=calories,
        protein_target=protein,
        fat_target=fat,
        carbs_target=carbs
    )
    await callback.message.edit_text(
        f"✅ Профиль сохранён!\n\n"
        f"Ваша дневная норма: {calories} ккал\n"
        f"Белки: {protein} г, Жиры: {fat} г, Углеводы: {carbs} г"
    )
    await callback.message.answer("Главное меню:", reply_markup=get_main_menu())
    await state.clear()

@dp.callback_query(ProfileStates.waiting_for_nutrient_choice, F.data == "nutrient_manual")
async def process_nutrient_manual_start(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Введите вашу дневную норму калорий (например, 2000):")
    await state.set_state(ManualNutrientStates.waiting_for_calories)

@dp.message(ManualNutrientStates.waiting_for_calories)
async def process_manual_calories(message: types.Message, state: FSMContext):
    try:
        calories = float(message.text)
        if calories < 1000 or calories > 5000:
            await message.answer("Пожалуйста, введите число от 1000 до 5000 ккал.")
            return
    except:
        await message.answer("Пожалуйста, введите корректное число.")
        return
    await state.update_data(manual_calories=calories)
    await message.answer("Введите количество белков в граммах (например, 100):")
    await state.set_state(ManualNutrientStates.waiting_for_protein)

@dp.message(ManualNutrientStates.waiting_for_protein)
async def process_manual_protein(message: types.Message, state: FSMContext):
    try:
        protein = float(message.text)
        if protein < 30 or protein > 300:
            await message.answer("Пожалуйста, введите число от 30 до 300 г.")
            return
    except:
        await message.answer("Пожалуйста, введите корректное число.")
        return
    await state.update_data(manual_protein=protein)
    await message.answer("Введите количество жиров в граммах (например, 70):")
    await state.set_state(ManualNutrientStates.waiting_for_fat)

@dp.message(ManualNutrientStates.waiting_for_fat)
async def process_manual_fat(message: types.Message, state: FSMContext):
    try:
        fat = float(message.text)
        if fat < 20 or fat > 200:
            await message.answer("Пожалуйста, введите число от 20 до 200 г.")
            return
    except:
        await message.answer("Пожалуйста, введите корректное число.")
        return
    await state.update_data(manual_fat=fat)
    await message.answer("Введите количество углеводов в граммах (например, 250):")
    await state.set_state(ManualNutrientStates.waiting_for_carbs)

@dp.message(ManualNutrientStates.waiting_for_carbs)
async def process_manual_carbs(message: types.Message, state: FSMContext):
    try:
        carbs = float(message.text)
        if carbs < 50 or carbs > 600:
            await message.answer("Пожалуйста, введите число от 50 до 600 г.")
            return
    except:
        await message.answer("Пожалуйста, введите корректное число.")
        return
    data = await state.get_data()
    # Сохраняем пользователя с ручными значениями
    db.create_user(
        telegram_id=message.from_user.id,
        goal=data['goal'],
        diet=data['diet'],
        allergies=data['allergies'],
        age=data['age'],
        weight=data['weight'],
        height=data['height'],
        gender=data['gender'],
        activity=data['activity'],
        calories_target=data['manual_calories'],
        protein_target=data['manual_protein'],
        fat_target=data['manual_fat'],
        carbs_target=carbs
    )
    await message.answer(
        f"✅ Профиль сохранён с ручными параметрами!\n\n"
        f"Ваша дневная норма: {data['manual_calories']} ккал\n"
        f"Белки: {data['manual_protein']} г, Жиры: {data['manual_fat']} г, Углеводы: {carbs} г"
    )
    await message.answer("Главное меню:", reply_markup=get_main_menu())
    await state.clear()

# ---------- Остальные хендлеры (профиль, инвентарь, меню) ----------
@dp.message(F.text == "👤 Мой профиль")
async def show_profile(message: types.Message):
    user = db.get_user(message.from_user.id)
    if not user:
        await message.answer("Профиль не найден. Нажмите /start для создания.")
        return
    text = (
        f"🎯 Цель: {user['goal']}\n"
        f"🥗 Диета: {user['diet']}\n"
        f"🚫 Аллергии: {user['allergies']}\n"
        f"📊 Антропометрия: {user['age']} лет, {user['weight']} кг, {user['height']} см, пол: {user['gender']}\n"
        f"🏃 Активность: {user['activity']}\n"
        f"🔥 Норма калорий: {user['calories_target']} ккал\n"
        f"🍗 БЖУ: {user['protein_target']}г / {user['fat_target']}г / {user['carbs_target']}г"
    )
    await message.answer(text)

@dp.message(F.text == "🧊 Мои продукты")
async def show_inventory(message: types.Message):
    items = db.get_inventory(message.from_user.id)
    if not items:
        text = "Холодильник пуст."
    else:
        text = "В холодильнике:\n" + "\n".join(
            [f"• {it['ingredient']}: {it['quantity']} {it['unit']} (из: {it['raw_text']})" for it in items]
        )
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Добавить", callback_data="inventory_add")],
        [InlineKeyboardButton(text="🗑 Очистить", callback_data="inventory_clear")]
    ])
    await message.answer(text, reply_markup=kb)

@dp.callback_query(F.data == "inventory_add")
async def inventory_add_start(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.answer("Введите продукты через запятую (например: 150 г курицы, 2 яйца, пучок укропа):")
    await state.set_state(InventoryStates.waiting_for_items)

@dp.message(InventoryStates.waiting_for_items)
async def inventory_process_text(message: types.Message, state: FSMContext):
    text = message.text
    pantry = PantryManager(db, message.from_user.id)
    candidates = pantry.parse_free_text(text)
    if not candidates:
        await message.answer("Не удалось распознать продукты. Попробуйте ещё раз.")
        return
    await state.update_data(candidates=candidates)
    response = "Проверьте распознанное:\n"
    for i, c in enumerate(candidates, 1):
        response += f"{i}. {c['raw']} → {c['canonical']} ({c['qty']} {c['unit']})\n"
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Подтвердить всё", callback_data="confirm_all")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")]
    ])
    await message.answer(response, reply_markup=kb)
    await state.set_state(InventoryStates.confirm_items)

@dp.callback_query(InventoryStates.confirm_items, F.data == "confirm_all")
async def inventory_confirm_all(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    candidates = data['candidates']
    pantry = PantryManager(db, callback.from_user.id)
    for cand in candidates:
        pantry.confirm_and_add(cand)
    await callback.message.edit_text("Продукты добавлены в холодильник!")
    await state.clear()

@dp.callback_query(F.data == "inventory_clear")
async def inventory_clear(callback: types.CallbackQuery):
    db.clear_inventory(callback.from_user.id)
    await callback.answer()
    await callback.message.edit_text("Холодильник очищен.")

@dp.message(F.text == "🍎 Составить меню")
async def menu_start(message: types.Message, state: FSMContext):
    await message.answer("Напишите ваши пожелания к меню (например: 'рацион на день, без молока'):")
    await state.set_state(MenuStates.waiting_for_query)

def split_text(text: str, max_length: int = 4096) -> List[str]:
    """Разбивает длинный текст на части, не превышающие max_length."""
    lines = text.split('\n')
    parts = []
    current_part = ""
    for line in lines:
        if len(current_part) + len(line) + 1 <= max_length:
            current_part += line + '\n'
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = line + '\n'
    if current_part:
        parts.append(current_part.strip())
    return parts

@dp.message(MenuStates.waiting_for_query)
async def process_menu_query(message: types.Message, state: FSMContext):
    query = message.text
    user_id = message.from_user.id
    user = db.get_user(user_id)
    if not user:
        await message.answer("Сначала заполните профиль (/start).")
        await state.clear()
        return

    await message.answer("⏳ Составляю рацион...")
    if USE_AGENT and agent:
        response = agent.run(user_id, query)
    else:
        profile = {
            'allergies': user['allergies'],
            'diet': user['diet']
        }
        meal_plan = rag.build_meal_plan(query, profile)
        response = rag.format_meal_plan(meal_plan)

    parts = split_text(response)
    for part in parts:
        await message.answer(part, parse_mode=None)
    await state.clear()

@dp.message(F.text == "🛒 Список покупок")
async def shopping_list(message: types.Message):
    await message.answer("Функция в разработке.")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
