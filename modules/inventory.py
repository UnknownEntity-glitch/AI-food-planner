# inventory.py - Модуль учёта домашних запасов (полная версия)

import re
import json
from collections import defaultdict, namedtuple
from difflib import get_close_matches
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict, Any, List
from datetime import datetime, timezone

Ingredient = namedtuple('Ingredient', ['raw', 'name', 'qty', 'unit', 'qty_base', 'unit_base', 'approx'])

SYNONYMS_FILE = Path('synonyms.json')
PANTRY_FILE = Path('pantry.json')

SEED_CANONICAL = [
    "куриная грудка", "курица", "филе куриное", "лук репчатый", "лук", "чеснок",
    "масло растительное", "масло оливковое", "рис", "гречка", "овсянка", "сыр",
    "молоко", "яйцо", "авокадо", "тунец консервированный", "тунец", "картофель",
    "морковь", "помидор", "огурец", "сахар", "соль", "перец чёрный молотый",
    "мука пшеничная", "мука", "йогурт", "кефир", "бекон", "паста", "макароны",
    "банан", "яблоко", "лимон", "капуста", "свёкла", "петрушка", "укроп",
    "сметана", "творог", "сливки", "говядина", "свинина", "баранина", "индейка",
    "рыба", "хлеб", "батон", "вода", "сода", "уксус", "корица", "кефир"
]

UNIT_SYNONYMS = {
    'g':        ['g', 'гр', 'гр.', 'г', 'gram', 'grams'],
    'kg':       ['kg', 'кг', 'килограмм', 'килограм', 'kg.'],
    'ml':       ['ml', 'мл', 'мл.', 'milliliter', 'millilitre'],
    'l':        ['l', 'л', 'литр', 'литров', 'литра'],
    'tsp':      ['tsp', 'ч.л', 'ч.л.', 'чл', 'ч/л', 'чайная_ложка', 'ч.ложка', 'ч л', 'ч'],
    'tbsp':     ['tbsp', 'ст.л', 'ст.л.', 'стл', 'ст.л.', 'столовая_ложка', 'столовая ложка', 'ст. ложка'],
    'cup':      ['cup', 'cups', 'стакан', 'чашка'],
    'pc':       ['pc', 'pcs', 'шт', 'штук', 'piece', 'pieces', 'шт.', 'шт'],
    'clove':    ['зубчик', 'зубчика', 'зубчиков', 'clove'],
    'can':      ['банка', 'банок', 'консерв', 'can'],
    'pack':     ['пачка', 'упак', 'пакет', 'pack'],
    'pinch':    ['щепотка', 'щепот'],
    'bunch':    ['пучок', 'пучка', 'bunch'],
    'head':     ['головка', 'головку', 'head'],
}

DEFAULT_QUANTITY_WORDS = {
    'зубчик': ('clove', 1),
    'зубчика': ('clove', 1),
    'зубчиков': ('clove', 1),
    'банка': ('can', 1),
    'банки': ('can', 1),
    'пачка': ('pack', 1),
    'пакетик': ('pack', 1),
    'пучок': ('bunch', 1),
    'пучка': ('bunch', 1),
    'головка': ('head', 1),
    'головки': ('head', 1),
    'кочан': ('head', 1),
    'кочана': ('head', 1),
    'щепотка': ('pinch', 1),
    'щепотки': ('pinch', 1),
    'буханка': ('pc', 1),
    'батон': ('pc', 1),
}

def _norm_alias(a: str) -> str:
    return str(a).strip().lower().replace('ё', 'е')

UNIT_MAP = {}
for k, vals in UNIT_SYNONYMS.items():
    for v in vals:
        UNIT_MAP[_norm_alias(v)] = k

CONV = {
    ('kg', 'g'):    1000.0,
    ('g', 'g'):     1.0,
    ('l', 'ml'):    1000.0,
    ('ml', 'ml'):   1.0,
    ('tbsp', 'ml'): 15.0,
    ('tsp', 'ml'):  5.0,
    ('cup', 'ml'):  240.0,
    ('pc', 'pc'):   1.0,
    ('clove', 'pc'): 1.0,
    ('can', 'pc'):  1.0,
    ('pack', 'pc'): 1.0,
    ('pinch', 'g'): 0.5,
    ('bunch', 'g'): 50.0,
    ('head', 'pc'): 1.0,
}

CONTAINER_TOKENS = ['банка', 'пачка', 'стакан', 'чашка', 'буханка', 'кочан', 'пучок', 'головка']
ADJECTIVE_TOKENS = ['свежий', 'мелкий', 'крупный', 'целый', 'маленький', 'большой', 'домашний']
LIQUID_KEYWORDS = ['молок', 'вода', 'кефир', 'сок', 'сливк', 'уксус']
VERBAL_QUANTITIES = {
    'половина': 0.5, 'пол': 0.5, 'половин': 0.5,
    'треть': 1/3, 'четверть': 0.25,
    'полтора': 1.5, 'пару': 2, 'пара': 2, 'несколько': 2,
    'полкило': 0.5, 'килограмм': 1.0, 'полкилограмма': 0.5,
}
VOLUME_WORDS = ['стакан', 'чашка', 'ложк', 'столовая', 'чайная', 'литр', 'мл']

def load_synonyms() -> Dict[str, str]:
    if SYNONYMS_FILE.exists():
        try:
            with open(SYNONYMS_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            return {normalize_text(k): normalize_text(v) for k, v in raw.items()}
        except:
            return {}
    return {}

SYNONYMS = load_synonyms()

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower().replace('ё', 'е')
    s = re.sub(r'[^0-9a-zа-яё\-\s\(\)]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

UNICODE_FRAC = {'½': '1/2', '⅓': '1/3', '¼': '1/4', '¾': '3/4'}

def replace_unicode_fractions(s: str) -> str:
    for k, v in UNICODE_FRAC.items():
        s = s.replace(k, v)
    return s

def parse_verbal_quantity(text: str) -> Tuple[Optional[float], str]:
    text_lower = text.lower().strip()
    for word, val in VERBAL_QUANTITIES.items():
        if text_lower.startswith(word):
            rest = text[len(word):].lstrip(' ,')
            return val, rest
    return None, text

def parse_qty(txt: Optional[str]) -> Optional[float]:
    if txt is None:
        return None
    s = replace_unicode_fractions(str(txt))
    s = s.strip().replace(',', '.')
    s = re.sub(r'(\d)([а-яa-z])', r'\1 \2', s)
    if re.search(r'[\-–]', s):
        parts = re.split(r'[\-–]', s)
        nums = [float(p) for p in parts if p.replace('.','',1).isdigit()]
        if nums:
            return sum(nums) / len(nums)
    m = re.match(r'^(\d+)\s+(\d+)/(\d+)$', s)
    if m:
        return int(m.group(1)) + int(m.group(2))/int(m.group(3))
    m = re.match(r'^(\d+)/(\d+)$', s)
    if m:
        return int(m.group(1))/int(m.group(2))
    m = re.search(r'\d+[\d\.,/]*', s)
    if m:
        tok = m.group(0).replace(',', '.')
        try:
            return float(tok)
        except:
            return None
    try:
        return float(s)
    except:
        return None

def normalize_unit_str(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    s = _norm_alias(str(u))
    if s in UNIT_MAP:
        return UNIT_MAP[s]
    cand = get_close_matches(s, list(UNIT_MAP.keys()), n=1, cutoff=0.85)
    if cand:
        return UNIT_MAP[cand[0]]
    if 'стак' in s or 'чаш' in s:
        return 'cup'
    if 'столов' in s or 'ст.' in s:
        return 'tbsp'
    if 'чайн' in s or 'ч.' in s:
        return 'tsp'
    if 'зуб' in s:
        return 'clove'
    if 'бан' in s:
        return 'can'
    if 'пач' in s or 'пак' in s:
        return 'pack'
    if 'пуч' in s:
        return 'bunch'
    if 'голов' in s:
        return 'head'
    if 'щепот' in s:
        return 'pinch'
    return None

def convert_to_base(qty: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], Optional[str], bool]:
    if qty is None:
        return (None, None, True)
    u = normalize_unit_str(unit) if unit else None
    if u is None:
        return (qty, None, True)
    approx = False
    if u in ('kg', 'g'):
        if u == 'kg':
            return (qty * CONV[('kg','g')], 'g', False)
        return (qty, 'g', False)
    if u in ('l', 'ml'):
        if u == 'l':
            return (qty * CONV[('l','ml')], 'ml', False)
        return (qty, 'ml', False)
    if u == 'tbsp':
        return (qty * CONV[('tbsp','ml')], 'ml', True)
    if u == 'tsp':
        return (qty * CONV[('tsp','ml')], 'ml', True)
    if u == 'cup':
        return (qty * CONV[('cup','ml')], 'ml', True)
    if u == 'pinch':
        return (qty * CONV[('pinch','g')], 'g', True)
    if u == 'bunch':
        return (qty * CONV[('bunch','g')], 'g', True)
    if u in ('pc', 'clove', 'can', 'pack', 'head'):
        return (qty, 'pc', False)
    return (qty, None, True)

def clean_canonical_name(name: str) -> str:
    if not name:
        return name
    s = normalize_text(name)
    tokens = s.split()
    filtered = [t for t in tokens if t not in CONTAINER_TOKENS and t not in ADJECTIVE_TOKENS]
    s = ' '.join(filtered)
    s = re.sub(r'[\(\)\.]', ' ', s)
    s = re.sub(r'\b[а-я]\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else name

def canonicalize(raw: str, vocab: Optional[Iterable[str]] = None) -> str:
    if raw is None:
        return ''
    n = normalize_text(raw)
    if n in SYNONYMS:
        return SYNONYMS[n]
    n = re.sub(r'\([^)]*\)', ' ', n)
    for tok in CONTAINER_TOKENS + ADJECTIVE_TOKENS:
        n = re.sub(r'\b' + re.escape(tok) + r'\b', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    if not n:
        return clean_canonical_name(raw)
    if vocab is None:
        vocab = SEED_CANONICAL
    vocab_norm = {normalize_text(v): v for v in vocab}
    if n in vocab_norm:
        return vocab_norm[n]
    for vnorm, vorig in vocab_norm.items():
        if vnorm and vnorm in n:
            return vorig
    tokens = [t for t in n.split() if len(t) > 2]
    best = None
    best_score = 0
    for t in tokens:
        cand = get_close_matches(t, list(vocab_norm.keys()), n=3, cutoff=0.72)
        if cand:
            score = len(t)
            if score > best_score:
                best_score = score
                best = vocab_norm[cand[0]]
    if best:
        return best
    cand_full = get_close_matches(n, list(vocab_norm.keys()), n=1, cutoff=0.70)
    if cand_full:
        return vocab_norm[cand_full[0]]
    return clean_canonical_name(n)

def extract_parenthetical_grams(text: str) -> Optional[Tuple[float, str]]:
    m = re.search(r'\(([^)]*?(?:г|гр|кг|мл|л)[^)]*?)\)', text)
    if not m:
        return None
    inside = m.group(1)
    m2 = re.search(r'(\d+[ \d\/\.,]*)\s*(г|гр|кг|мл|л)', inside)
    if m2:
        q = parse_qty(m2.group(1))
        u = m2.group(2)
        return (q, u)
    return None

UNIT_PATTERN = r'(?:г|гр|кг|мл|л|шт|ч\.л|ч\.л\.|ст\.л|ст\.л\.|tbsp|tsp|cup|cups|piece|зубчик|банка|пачка|упак|щепотка|пучок|головка|буханка|кочан|веточка|долька)'
QUANTITY_REGEX = r'(?P<qty>\d+[ \d\/\.,\-–]*)'
PATTERN = re.compile(rf'^\s*{QUANTITY_REGEX}\s*(?P<unit>{UNIT_PATTERN})?(?:\s+|$)(?P<name>.*)$', re.IGNORECASE)

def parse_ingredient_line(line: str) -> Optional[Ingredient]:
    if not line:
        return None
    raw = str(line).strip()
    raw = replace_unicode_fractions(raw)

    verbal_qty, rest_after_verbal = parse_verbal_quantity(raw)
    if verbal_qty is not None:
        if rest_after_verbal:
            name_candidate = rest_after_verbal.strip()
            if name_candidate:
                canon = canonicalize(name_candidate)
                qty_base, unit_base, approx = convert_to_base(verbal_qty, None)
                return Ingredient(raw, canon, verbal_qty, None, qty_base, unit_base, True)
        return None

    parent = extract_parenthetical_grams(raw)
    if parent:
        qty, unit = parent
        name_no_paren = re.sub(r'\([^)]*\)', '', raw).strip()
        m = PATTERN.match(name_no_paren)
        if m and m.group('name'):
            name_part = m.group('name').strip()
        else:
            name_part = re.sub(r'^\s*\d+[ \d\/\.,\-–]*\s*', '', name_no_paren).strip()
        if not name_part:
            name_part = name_no_paren
        canon = canonicalize(name_part)
        qty_base, unit_base, approx = convert_to_base(qty, unit)
        return Ingredient(raw, canon, qty, unit, qty_base, unit_base, approx)

    m = PATTERN.match(raw)
    if m:
        qty_raw = m.group('qty').strip()
        unit_raw = m.group('unit') or ''
        name_raw = m.group('name').strip() if m.group('name') else ''
        name_raw = re.sub(r'^(of|для|на)\s+', '', name_raw, flags=re.IGNORECASE).strip()
        qty = parse_qty(qty_raw)
        unit = normalize_unit_str(unit_raw) if unit_raw else None
        name_norm = normalize_text(name_raw)
        if unit is None and qty is not None:
            if any(kw in name_norm for kw in LIQUID_KEYWORDS):
                unit = 'l'
        if not name_raw and unit_raw:
            after = raw[len(qty_raw):].lstrip()
            if after.lower().startswith(unit_raw.lower()):
                after = after[len(unit_raw):].lstrip()
            name_raw = after
        if name_raw:
            words = name_raw.split()
            if words and normalize_unit_str(words[0]) is not None:
                words = words[1:]
                name_raw = ' '.join(words)
        canon = canonicalize(name_raw) if name_raw else "неизвестный_ингредиент"
        qty_base, unit_base, approx = convert_to_base(qty, unit)
        return Ingredient(raw, canon, qty, unit, qty_base, unit_base, approx)

    m2 = re.search(r'(\d+[ \d\/\.,]*)([а-яa-z]+)', raw)
    if m2:
        qty_part = m2.group(1)
        rest = m2.group(2) + raw[m2.end():]
        qty = parse_qty(qty_part)
        found_unit = None
        for unit_alias in UNIT_MAP.keys():
            if rest.lower().startswith(unit_alias):
                found_unit = unit_alias
                rest = rest[len(unit_alias):].lstrip()
                break
        unit = normalize_unit_str(found_unit) if found_unit else None
        name_raw = rest.strip()
        if name_raw:
            canon = canonicalize(name_raw)
            qty_base, unit_base, approx = convert_to_base(qty, unit)
            return Ingredient(raw, canon, qty, unit, qty_base, unit_base, approx)

    m3 = re.search(r'(\d+[ \d\/\.,]*)\s*(' + UNIT_PATTERN + r')?(?:\s+|$)', raw)
    if m3:
        qty = parse_qty(m3.group(1))
        unit = normalize_unit_str(m3.group(2)) if m3.group(2) else None
        name_part = raw.replace(m3.group(0), '').strip(' ,;.-')
        if not name_part:
            name_part = raw
        canon = canonicalize(name_part)
        qty_base, unit_base, approx = convert_to_base(qty, unit)
        return Ingredient(raw, canon, qty, unit, qty_base, unit_base, approx)

    words = raw.lower().split()
    for w in words:
        if w in DEFAULT_QUANTITY_WORDS:
            unit_word, default_qty = DEFAULT_QUANTITY_WORDS[w]
            name_clean = re.sub(r'\b' + re.escape(w) + r'\b', '', raw, flags=re.IGNORECASE).strip()
            if not name_clean:
                name_clean = raw
            canon = canonicalize(name_clean)
            qty_base, unit_base, approx = convert_to_base(default_qty, unit_word)
            return Ingredient(raw, canon, default_qty, unit_word, qty_base, unit_base, approx)

    canon = canonicalize(raw)
    return Ingredient(raw, canon, None, None, None, None, True)


# ---------- Класс для работы с БД ----------
class PantryManager:
    def __init__(self, db, user_id: int):
        self.db = db
        self.user_id = user_id

    def parse_free_text(self, text: str) -> List[Dict]:
        parts = re.split(r',|\band\b|\bи\b|\b&\b|;', text)
        candidates = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            ing = parse_ingredient_line(part)
            if ing is None:
                continue
            candidates.append({
                'raw': part,
                'canonical': ing.name,
                'qty': ing.qty,
                'unit': ing.unit,
                'qty_base': ing.qty_base,
                'unit_base': ing.unit_base,
                'approx': ing.approx
            })
        return candidates

    def confirm_and_add(self, entry: Dict):
        self.db.add_inventory_item(
            user_id=self.user_id,
            ingredient=entry['canonical'],
            quantity=entry['qty_base'] or entry['qty'] or 1.0,
            unit=entry['unit_base'] or entry['unit'] or 'pc',
            raw_text=entry['raw']
        )

    def get_available(self) -> Dict[str, float]:
        items = self.db.get_inventory(self.user_id)
        total = defaultdict(float)
        for it in items:
            total[it['ingredient']] += it['quantity']
        return dict(total)

    def can_make(self, recipe_ingredients: List[Dict]) -> bool:
        available = self.get_available()
        needed = defaultdict(float)
        for ing in recipe_ingredients:
            needed[ing['name']] += ing['qty_base']
        for name, qty in needed.items():
            if available.get(name, 0) < qty - 1e-6:
                return False
        return True

    def consume(self, recipe_ingredients: List[Dict]) -> Dict:
        items = self.db.get_inventory(self.user_id)
        by_ing = defaultdict(list)
        for it in items:
            by_ing[it['ingredient']].append(it)

        consumed = []
        missing = []

        for need in recipe_ingredients:
            name = need['name']
            need_qty = need['qty_base']
            if name not in by_ing:
                missing.append(need)
                continue
            entries = by_ing[name]
            remaining = need_qty
            for entry in entries[:]:
                if remaining <= 0:
                    break
                take = min(entry['quantity'], remaining)
                entry['quantity'] -= take
                remaining -= take
                consumed.append({
                    'ingredient': name,
                    'taken': take,
                    'from_raw': entry['raw']
                })
                if entry['quantity'] <= 1e-6:
                    self.db.remove_inventory_item(self.user_id, entry['raw'])
            if remaining > 0:
                missing.append({'name': name, 'qty_base': remaining, 'unit_base': need['unit_base']})

        return {'consumed': consumed, 'missing': missing}