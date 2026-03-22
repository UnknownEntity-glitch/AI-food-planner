# rag.py (полная версия с методом get_recipe_by_category)

import pandas as pd
import numpy as np
import faiss
import zipfile
import os
import re
import random
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Скачиваем стоп-слова
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class RecipeRAG:
    def __init__(self, data_dir: str, zip_path: str):
        self.data_dir = data_dir
        self.zip_path = zip_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.stopwords = set(stopwords.words('russian'))
        self.df = None
        self.index = None
        self._load_or_build()

    def _load_or_build(self):
        index_file = os.path.join(self.data_dir, 'faiss.index')
        df_file = os.path.join(self.data_dir, 'recipes.parquet')
        if os.path.exists(index_file) and os.path.exists(df_file):
            self.index = faiss.read_index(index_file)
            self.df = pd.read_parquet(df_file)
            print(f"✅ Загружен индекс с {len(self.df)} рецептами")
        else:
            print("🔄 Индекс не найден, строим из архива...")
            self._build_from_zip()
            faiss.write_index(self.index, index_file)
            self.df.to_parquet(df_file)
            print(f"✅ Индекс сохранён, {len(self.df)} рецептов")

    def _build_from_zip(self):
        """Строит индекс из архива recipes.zip (код из ячейки 4 оригинального ноутбука)."""
        # --- 1. Распаковка архива ---
        print("Загрузите файл recipes.zip (он должен быть в data_dir)")
        # Вместо загрузки через files.upload() просто проверяем наличие файла
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Файл {self.zip_path} не найден. Поместите архив recipes.zip в папку {self.data_dir}")

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.data_dir, 'recipes_temp'))
        print("✅ Архив распакован в папку 'recipes_temp'")

        # --- 2. Диагностика содержимого (необязательно) ---
        txt_files = []
        for root, dirs, files in os.walk(os.path.join(self.data_dir, 'recipes_temp')):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        print(f"Найдено .txt файлов: {len(txt_files)}")

        if not txt_files:
            raise RuntimeError("Нет .txt файлов! Проверьте содержимое архива.")

        # --- 3. Функция парсинга рецепта (из оригинального файла) ---
        def parse_recipe_block(block):
            lines = block.strip().split('\n')
            if not lines[0].startswith('Название:'):
                return None
            title = lines[0].replace('Название:', '').strip()

            ingredients_start = None
            recipe_start = None
            advice_start = None
            for i, line in enumerate(lines):
                if line.strip() == 'Ингредиенты':
                    ingredients_start = i
                elif line.strip() == 'Рецепт':
                    recipe_start = i
                elif line.strip() == 'Совет':
                    advice_start = i

            ingredients_list = []
            if ingredients_start is not None:
                end_idx = recipe_start if recipe_start is not None else len(lines)
                for j in range(ingredients_start+1, end_idx):
                    line = lines[j].strip()
                    if line and not line.startswith('Рецепт') and not line.startswith('Совет'):
                        ingredients_list.append(line)
            ingredients_text = ' '.join(ingredients_list)

            instructions = ''
            if recipe_start is not None:
                end_idx = advice_start if advice_start is not None else len(lines)
                instructions = '\n'.join(lines[recipe_start+1:end_idx]).strip()

            advice = ''
            if advice_start is not None:
                advice = '\n'.join(lines[advice_start+1:]).strip()

            # Пытаемся найти калорийность и БЖУ
            calories = None
            protein = fat = carbs = None
            pattern = r'Калорийность на 100 грамм:\s*([\d.]+)\s*ккал.*?Б/Ж/У:\s*([\d.]+)/([\d.]+)/([\d.]+)'
            match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
            if match:
                calories = float(match.group(1))
                protein = float(match.group(2))
                fat = float(match.group(3))
                carbs = float(match.group(4))

            return {
                'title': title,
                'ingredients_list': ingredients_list,
                'ingredients_text': ingredients_text,
                'instructions': instructions,
                'advice': advice,
                'calories': calories,
                'protein': protein,
                'fat': fat,
                'carbs': carbs
            }

        # --- 4. Парсинг всех файлов ---
        all_recipes = []
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            parts = re.split(r'(?=^Название:)', content, flags=re.MULTILINE)
            for block in parts:
                if block.strip():
                    parsed = parse_recipe_block(block)
                    if parsed:
                        all_recipes.append(parsed)

        print(f"✅ Загружено {len(all_recipes)} рецептов")
        if len(all_recipes) == 0:
            raise RuntimeError("Не удалось распарсить ни одного рецепта. Проверьте формат файлов.")

        # --- 5. Создание DataFrame ---
        self.df = pd.DataFrame(all_recipes)
        self.df['id'] = range(len(self.df))
        self.df['text_for_embedding'] = self.df['title'] + ' ' + self.df['ingredients_text']

        # --- 6. Добавление меток категорий (из ячейки 6) ---
        print("Обработка меток категорий из рецептов...")
        def extract_meal_type(instructions):
            if not isinstance(instructions, str):
                return instructions, None
            lines = instructions.strip().split('\n')
            if not lines:
                return instructions, None
            last_line = lines[-1].strip()
            if last_line.startswith('(') and last_line.endswith(')'):
                new_instructions = '\n'.join(lines[:-1]).strip()
                meal_type = last_line[1:-1].strip()
                return new_instructions, meal_type
            return instructions, None

        new_instructions = []
        meal_types = []
        for idx, row in self.df.iterrows():
            instr, mtype = extract_meal_type(row['instructions'])
            new_instructions.append(instr)
            meal_types.append(mtype)

        self.df['instructions'] = new_instructions
        self.df['meal_type'] = meal_types

        # Функция для fallback-категоризации (упрощённая)
        def categorize_recipe(row):
            title = str(row['title']).lower()
            if any(word in title for word in ['завтрак', 'каша', 'омлет', 'яичница', 'сырник']):
                return 'завтрак'
            elif any(word in title for word in ['салат']):
                return 'салат'
            elif any(word in title for word in ['суп', 'борщ', 'щи', 'уха', 'солянка', 'бульон']):
                return 'суп'
            elif any(word in title for word in ['десерт', 'торт', 'пирожное', 'печенье', 'кекс', 'сладкое']):
                return 'десерт'
            else:
                return 'основное блюдо'

        def get_meal_type_fallback(row):
            if pd.notna(row['meal_type']):
                return row['meal_type']
            return categorize_recipe(row)

        self.df['meal_type'] = self.df.apply(get_meal_type_fallback, axis=1)
        print("Распределение по типам блюд:")
        print(self.df['meal_type'].value_counts())

        # --- 7. Добавление маркеров (диабет, без глютена, кето) ---
        print("Добавляем маркеры для рецептов...")
        def has_marker(text, marker_words):
            if not isinstance(text, str):
                return False
            text_lower = text.lower()
            return any(marker in text_lower for marker in marker_words)

        self.df['is_diabetic'] = self.df['title'].apply(lambda x: has_marker(x, ['диабет', 'для диабетиков'])) | \
                                 self.df['instructions'].apply(lambda x: has_marker(x, ['диабет', 'для диабетиков']))
        self.df['is_gluten_free'] = self.df['title'].apply(lambda x: has_marker(x, ['без глютена', 'безглютен'])) | \
                                    self.df['instructions'].apply(lambda x: has_marker(x, ['без глютена', 'безглютен']))
        self.df['is_keto'] = self.df['title'].apply(lambda x: has_marker(x, ['кето'])) | \
                             self.df['instructions'].apply(lambda x: has_marker(x, ['кето']))
        print("Маркеры добавлены.")

        # --- 8. Вычисление эмбеддингов и построение индекса ---
        embeddings = self.embedder.encode(self.df['text_for_embedding'].tolist(), show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # inner product = cosine for normalized vectors
        self.index.add(embeddings)

        # --- 9. Очистка временной папки ---
        import shutil
        shutil.rmtree(os.path.join(self.data_dir, 'recipes_temp'))

    def search(self, query: str, top_k: int = 100) -> pd.DataFrame:
        """Векторный поиск по запросу."""
        emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        distances, indices = self.index.search(emb, top_k)
        results = self.df.iloc[indices[0]].copy()
        results['score'] = distances[0]
        return results

    # ---------- Методы для составления рациона (из оригинального файла) ----------
    def _extract_allergens_and_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        q = query.lower()
        allergen_patterns = [
            r'(?:аллергия|аллерги[яию])\s+на\s+([^\.;,]+)',
            r'не\s+переношу\s+([^\.;,]+)',
            r'исключ(?:ить|ая)\s+([^\.;,]+)',
            r'\bбез\s+(\w+)',
            r'\bне\s+содержит\s+(\w+)',
        ]
        allergens_raw = []
        for pattern in allergen_patterns:
            matches = re.findall(pattern, q)
            for match in matches:
                parts = re.split(r'[,\s]+(?:и|или)?\s+', match)
                allergens_raw.extend([p.strip() for p in parts if p.strip()])

        allergens = list(set([a for a in allergens_raw if len(a) > 2]))

        q_clean = q
        for a in allergens:
            q_clean = re.sub(r'\b' + re.escape(a) + r'\b', '', q_clean)

        for phrase in ['аллергия', 'аллергии', 'аллергию', 'не переношу', 'исключить', 'исключая',
                       'составь рацион', 'рацион на день', 'из 3 блюд', 'пожалуйста', 'меню']:
            q_clean = q_clean.replace(phrase, '')

        words = word_tokenize(q_clean, language='russian')
        keywords = [w for w in words if w.isalpha() and w not in self.stopwords and len(w) > 2]

        return allergens, keywords

    def _extract_dietary_preferences(self, query: str) -> Dict[str, bool]:
        q = query.lower()
        prefs = {
            'vegetarian': False,
            'vegan': False,
            'raw': False,
            'diabetic': False,
            'gluten_free': False,
            'keto': False,
            'calorie_mentioned': False
        }
        if any(word in q for word in ['вегетарианец', 'вегетарианка', 'вегетарианство']):
            prefs['vegetarian'] = True
        if any(word in q for word in ['веган', 'веганство']):
            prefs['vegan'] = True
            prefs['vegetarian'] = True
        if any(word in q for word in ['сыроед', 'сыроедение', 'raw']):
            prefs['raw'] = True
        if any(word in q for word in ['диабет', 'диабетик']):
            prefs['diabetic'] = True
        if any(word in q for word in ['без глютена', 'безглютеновая', 'глютен']):
            prefs['gluten_free'] = True
        if any(word in q for word in ['кето', 'кетогенная']):
            prefs['keto'] = True
        if any(word in q for word in ['калорий', 'ккал', 'калории']):
            prefs['calorie_mentioned'] = True
        return prefs

    def _passes_dietary(self, row, prefs: Dict[str, bool]) -> bool:
        ing = row['ingredients_text'].lower()
        instr = row['instructions'].lower()

        meat_keywords = ['куриц', 'курин', 'кур', 'говяд', 'свин', 'баран', 'телят', 'индейк', 'утк', 'утин',
                         'гус', 'кролик', 'печен', 'сердц', 'фарш', 'свинин', 'телятин', 'конин',
                         'курятин', 'индюш', 'индюшатин', 'филе', 'вырезк', 'ребр', 'кострец',
                         'грудинк', 'бедр', 'шейк', 'грудк', 'лопатк', 'рульк', 'хрящ', 'бекон',
                         'окорок', 'окорочк', 'подбедерок', 'ножк', 'бедрышк', 'овч', 'овеч',
                         'карпаччо', 'хамон']
        fish_keywords = ['рыб', 'тунец', 'тунц', 'лосос', 'форел', 'семг', 'креветк', 'креветоч',
                         'кальмар', 'миди', 'морепродукт', 'лобстер', 'устри']
        dairy_keywords = ['молок', 'кефир', 'йогурт', 'сметан', 'сливк', 'творог', 'сыр',
                          'простокваш', 'ряженк', 'сливочн', 'пармезан', 'камамбер', 'эмментал',
                          'чеддер', 'филадельфи', 'сулугуни', 'маасдам', 'халуми', 'фет', 'брынз',
                          'дор блю', 'горгонзол', 'грюйер', 'майонез']
        egg_keywords = ['яйц', 'яичн', 'белок', 'желток']
        honey_keywords = ['мёд', 'мед']

        if prefs.get('vegetarian') or prefs.get('vegan'):
            if any(kw in ing for kw in meat_keywords) or any(kw in ing for kw in fish_keywords):
                return False
        if prefs.get('vegan'):
            if any(kw in ing for kw in dairy_keywords) or any(kw in ing for kw in egg_keywords) or any(kw in ing for kw in honey_keywords):
                return False
        if prefs.get('raw'):
            cooking_keywords = ['жар', 'вар', 'печ', 'запека', 'туш', 'кипя', 'готов', 'пассер', 'бланш', 'прокал']
            if any(kw in instr for kw in cooking_keywords):
                return False
        return True

    def _is_unwanted_dish(self, title: str) -> bool:
        unwanted_keywords = [
            'хлеб', 'батон', 'булка', 'лепешка', 'лаваш', 'соус',
            'закуска', 'ассорти', 'багет', 'чиабатта'
        ]
        title_lower = title.lower()
        if any(word in title_lower for word in ['бутерброд', 'сэндвич', 'тост', 'брускетта']):
            return False
        return any(kw in title_lower for kw in unwanted_keywords)

    def build_meal_plan(self, query: str, user_profile: Optional[Dict] = None) -> Dict:
        """
        Составляет рацион на день.
        user_profile может содержать 'allergies' (строка) и 'diet' (строка).
        Возвращает словарь с категориями: 'breakfast', 'lunch', 'dinner', 'dessert' и т.д.
        Каждый элемент — словарь с ключами 'title', 'ingredients_list', 'instructions', и т.д.
        """
        allergens_from_query, keywords = self._extract_allergens_and_keywords(query)
        prefs = self._extract_dietary_preferences(query)

        if user_profile:
            if user_profile.get('allergies'):
                profile_allergens = [a.strip() for a in user_profile['allergies'].split(',') if a.strip()]
                allergens_from_query.extend(profile_allergens)
            diet = user_profile.get('diet', '').lower()
            if 'вегетариан' in diet:
                prefs['vegetarian'] = True
            if 'веган' in diet:
                prefs['vegan'] = True
            if 'кето' in diet:
                prefs['keto'] = True

        candidates_df = self.search(query, top_k=500)
        if candidates_df.empty:
            return {}

        if allergens_from_query:
            mask = np.ones(len(candidates_df), dtype=bool)
            for idx, row in candidates_df.iterrows():
                for allerg in allergens_from_query:
                    if allerg.lower() in row['ingredients_text'].lower():
                        mask[idx] = False
                        break
            candidates_df = candidates_df[mask].copy()

        if any(prefs.get(k) for k in ['vegetarian', 'vegan', 'raw']):
            mask = np.ones(len(candidates_df), dtype=bool)
            for idx, row in candidates_df.iterrows():
                if not self._passes_dietary(row, prefs):
                    mask[idx] = False
            candidates_df = candidates_df[mask].copy()

        unwanted_mask = candidates_df['title'].apply(self._is_unwanted_dish)
        candidates_df = candidates_df[~unwanted_mask].copy()

        if candidates_df.empty:
            return {}

        keyword_scores = []
        for _, row in candidates_df.iterrows():
            full_text = (row['title'] + ' ' + row['ingredients_text'] + ' ' + row['instructions']).lower()
            score = 0
            for kw in keywords:
                occurrences = full_text.count(kw.lower())
                score += occurrences
            keyword_scores.append(score)

        max_keyword = max(keyword_scores) if keyword_scores else 1
        if max_keyword > 0:
            keyword_scores = [s / max_keyword for s in keyword_scores]
        else:
            keyword_scores = [0] * len(keyword_scores)

        candidates_df['keyword_score'] = keyword_scores
        candidates_df['combined_score'] = candidates_df['score'] + 0.3 * candidates_df['keyword_score']

        # Добавляем бонусы за маркеры (как в оригинале)
        bonus = 0.2
        if prefs.get('diabetic'):
            candidates_df.loc[candidates_df['is_diabetic'], 'combined_score'] += bonus
        if prefs.get('gluten_free'):
            candidates_df.loc[candidates_df['is_gluten_free'], 'combined_score'] += bonus
        if prefs.get('keto'):
            candidates_df.loc[candidates_df['is_keto'], 'combined_score'] += 0.4
            if 'carbs' in candidates_df.columns:
                known_carbs = candidates_df['carbs'].notna()
                low_carbs = known_carbs & (candidates_df['carbs'] <= 10)
                candidates_df.loc[low_carbs, 'combined_score'] += 0.2
                high_carbs = known_carbs & (candidates_df['carbs'] > 10)
                candidates_df.loc[high_carbs, 'combined_score'] -= 0.5

        if prefs.get('calorie_mentioned') and 'calories' in candidates_df.columns:
            candidates_df.loc[candidates_df['calories'].notna(), 'combined_score'] += 0.2

        candidates_df = candidates_df.sort_values('combined_score', ascending=False)

        meal_plan = {}

        # Завтрак
        breakfast = candidates_df[candidates_df['title'].str.contains('завтрак|каша|омлет|яичница|сырники', case=False, na=False)]
        if not breakfast.empty:
            meal_plan['breakfast'] = breakfast.iloc[0].to_dict()

        # Обед: салат, суп, основное
        salad = candidates_df[candidates_df['title'].str.contains('салат', case=False, na=False)]
        if not salad.empty:
            meal_plan['salad'] = salad.iloc[0].to_dict()

        soup = candidates_df[candidates_df['title'].str.contains('суп|борщ|щи|уха|солянка|бульон', case=False, na=False)]
        if not soup.empty:
            meal_plan['soup'] = soup.iloc[0].to_dict()

        main = candidates_df[~candidates_df['title'].str.contains('салат|суп|завтрак|десерт', case=False, na=False)]
        if not main.empty:
            meal_plan['main'] = main.iloc[0].to_dict()

        # Ужин (если есть ещё одно основное)
        if len(main) > 1:
            meal_plan['dinner'] = main.iloc[1].to_dict()
        elif not main.empty:
            meal_plan['dinner'] = main.iloc[0].to_dict()  # если только одно, повторим

        # Десерт
        dessert = candidates_df[candidates_df['title'].str.contains('десерт|сладкое|выпечка|пирожное|торт|печенье|кекс', case=False, na=False)]
        if not dessert.empty:
            meal_plan['dessert'] = dessert.iloc[0].to_dict()

        return meal_plan

    def format_meal_plan(self, meal_plan: Dict) -> str:
        """Форматирует рацион в читаемый текст (аналог print_meal_plan)."""
        if not meal_plan:
            return "К сожалению, не удалось составить рацион. Попробуйте изменить запрос."

        lines = []
        lines.append("Выбранные рецепты:")

        # Завтрак
        lines.append("\n1. ЗАВТРАК:")
        if 'breakfast' in meal_plan:
            r = meal_plan['breakfast']
            lines.append(f"   {r['title']}")
            lines.append("   Ингредиенты:")
            for ing in r.get('ingredients_list', []):
                lines.append(f"     - {ing}")
            lines.append("   Приготовление:")
            for line in r.get('instructions', '').split('\n'):
                lines.append(f"     {line}")
        else:
            lines.append("   (не найдено подходящих рецептов)")

        # Обед
        lines.append("\n2. ОБЕД:")
        if 'salad' in meal_plan:
            lines.append("   а) Салат/закуска:")
            r = meal_plan['salad']
            lines.append(f"      {r['title']}")
            lines.append("      Ингредиенты:")
            for ing in r.get('ingredients_list', []):
                lines.append(f"        - {ing}")
            lines.append("      Приготовление:")
            for line in r.get('instructions', '').split('\n'):
                lines.append(f"        {line}")
        else:
            lines.append("   а) Салат/закуска: (не найдено)")

        if 'soup' in meal_plan:
            lines.append("   б) Суп:")
            r = meal_plan['soup']
            lines.append(f"      {r['title']}")
            lines.append("      Ингредиенты:")
            for ing in r.get('ingredients_list', []):
                lines.append(f"        - {ing}")
            lines.append("      Приготовление:")
            for line in r.get('instructions', '').split('\n'):
                lines.append(f"        {line}")
        else:
            lines.append("   б) Суп: (не найдено)")

        if 'main' in meal_plan:
            lines.append("   в) Основное блюдо:")
            r = meal_plan['main']
            lines.append(f"      {r['title']}")
            lines.append("      Ингредиенты:")
            for ing in r.get('ingredients_list', []):
                lines.append(f"        - {ing}")
            lines.append("      Приготовление:")
            for line in r.get('instructions', '').split('\n'):
                lines.append(f"        {line}")
        else:
            lines.append("   в) Основное блюдо: (не найдено)")

        # Ужин
        lines.append("\n3. УЖИН:")
        if 'dinner' in meal_plan:
            r = meal_plan['dinner']
            lines.append(f"   {r['title']}")
            lines.append("   Ингредиенты:")
            for ing in r.get('ingredients_list', []):
                lines.append(f"     - {ing}")
            lines.append("   Приготовление:")
            for line in r.get('instructions', '').split('\n'):
                lines.append(f"     {line}")
        else:
            lines.append("   (не найдено)")

        if 'dessert' in meal_plan:
            lines.append("\n4. ДЕСЕРТ:")
            r = meal_plan['dessert']
            lines.append(f"   {r['title']}")
            lines.append("   Ингредиенты:")
            for ing in r.get('ingredients_list', []):
                lines.append(f"     - {ing}")
            lines.append("   Приготовление:")
            for line in r.get('instructions', '').split('\n'):
                lines.append(f"     {line}")

        lines.append("\n" + "="*60)
        return "\n".join(lines)

    # ---------- НОВЫЙ МЕТОД: поиск рецепта по категории ----------
    def get_recipe_by_category(self, query: str, category: str, top_k: int = 100) -> Optional[Dict]:
        """
        Поиск рецепта по запросу с обязательной фильтрацией по категории.
        Возвращает словарь с данными рецепта или None.
        """
        results = self.search(query, top_k=top_k)
        if results.empty:
            return None

        # Фильтрация по точному совпадению категории
        cat_filter = results['meal_type'].str.lower() == category.lower()
        filtered = results[cat_filter]

        # Если нет точных совпадений, ищем частичное вхождение
        if filtered.empty:
            filtered = results[results['meal_type'].str.lower().str.contains(category.lower(), na=False)]

        # Если и частичных нет, берём лучший результат без фильтрации (но с проверкой, что категория подходит)
        if filtered.empty:
            # Попробуем взять топ-1, если его категория подходит под категорию
            best = results.iloc[0]
            if best['meal_type'] and category.lower() in best['meal_type'].lower():
                return best.to_dict()
            else:
                return None

        best = filtered.iloc[0]
        return best.to_dict()