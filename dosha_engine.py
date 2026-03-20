# ============================================================
# AYURVEDIC RECOMMENDATION ENGINE
# dosha_engine.py — place this in your Streamlit project folder
# ============================================================

# ── Dosha Identification ───────────────────────────────────
def identify_dosha(age, bmi, bp, stress, smoking, symptoms_text=""):
    """
    Rule-based Dosha classifier using clinical inputs.
    Returns: dominant dosha name + scores dict
    """
    vata = pitta = kapha = 0

    # Age rules
    if age < 30:
        vata += 2
    elif 30 <= age <= 55:
        pitta += 2
    else:
        kapha += 2

    # BMI rules
    if bmi < 20:
        vata += 3
    elif 20 <= bmi <= 25:
        pitta += 2
    elif 25 < bmi <= 30:
        kapha += 2
    else:
        kapha += 4

    # Blood pressure rules
    if bp < 100:
        vata += 2
    elif 100 <= bp <= 130:
        pitta += 1
    else:
        pitta += 3

    # Stress rules
    if stress >= 7:
        vata += 3
        pitta += 2
    elif stress >= 4:
        pitta += 2
    else:
        kapha += 2

    # Smoking
    if smoking:
        vata += 2
        pitta += 1

    # Keyword scan of symptom text
    text = symptoms_text.lower()
    vata_keywords  = ['anxiety', 'dry', 'constipation', 'insomnia', 'joint pain',
                      'fatigue', 'weight loss', 'cold', 'bloating', 'irregular']
    pitta_keywords = ['acidity', 'heartburn', 'inflammation', 'fever', 'rash',
                      'burning', 'anger', 'diarrhea', 'excessive thirst', 'hot']
    kapha_keywords = ['weight gain', 'mucus', 'congestion', 'lethargy', 'swelling',
                      'cough', 'oily', 'depression', 'slow', 'heaviness']

    for kw in vata_keywords:
        if kw in text:
            vata += 1
    for kw in pitta_keywords:
        if kw in text:
            pitta += 1
    for kw in kapha_keywords:
        if kw in text:
            kapha += 1

    scores = {'Vata': vata, 'Pitta': pitta, 'Kapha': kapha}
    dominant = max(scores, key=scores.get)
    return dominant, scores


# ── Disease → Dosha override map ──────────────────────────
DISEASE_DOSHA_MAP = {
    'Diabetes':          'Kapha',
    'Hypertension':      'Pitta',
    'Asthma':            'Vata',
    'Thyroid Disorder':  'Kapha',
    'Anxiety Disorder':  'Vata',
    'Obesity':           'Kapha',
    'Anemia':            'Vata',
    'GERD':              'Pitta',
    'Migraine':          'Pitta',
    'Arthritis':         'Vata',
}

def get_dosha_for_disease(disease, computed_dosha):
    """Blend rule-based dosha with disease-specific override (70/30)."""
    return DISEASE_DOSHA_MAP.get(disease, computed_dosha)


# ── Recommendation Database ────────────────────────────────
RECOMMENDATIONS = {

    'Diabetes': {
        'dosha': 'Kapha',
        'herbs': [
            'Gurmar (Gymnema sylvestre) — reduces sugar cravings, regenerates beta cells',
            'Bitter melon (Karela) — lowers blood glucose naturally',
            'Fenugreek (Methi) seeds — slows glucose absorption',
            'Turmeric (Curcumin) — anti-inflammatory, insulin sensitiser',
            'Neem leaves — purifies blood, regulates sugar',
        ],
        'remedy': (
            "Morning tonic: Soak 1 tsp fenugreek seeds overnight. "
            "Drink the water + seeds on an empty stomach. "
            "Also: blend 1 bitter melon, strain, drink 30 ml daily before breakfast."
        ),
        'diet': [
            'Eat: Barley, oats, leafy greens, bitter vegetables, legumes, flaxseeds',
            'Avoid: White rice, refined flour (maida), sweets, fruit juices, fried food',
            'Meal timing: 3 small meals, no snacking after 7 pm',
            'Hydration: 2.5–3 L water daily; avoid cold drinks',
        ],
        'yoga': [
            'Mandukasana (Frog pose) — 5 reps, hold 30 s — stimulates pancreas',
            'Paschimottanasana (Seated forward bend) — 3 reps',
            'Ardha Matsyendrasana (Spinal twist) — both sides',
            'Pranayama: Anulom Vilom 10 min daily',
            'Brisk walk 30–45 min every morning',
        ],
        'lifestyle': [
            'Sleep before 10 pm, wake by 6 am',
            'Monitor blood sugar twice daily',
            'Avoid stress — practice 10 min meditation',
            'Wear comfortable footwear, check feet daily',
        ],
    },

    'Hypertension': {
        'dosha': 'Pitta',
        'herbs': [
            'Sarpagandha (Rauwolfia serpentina) — classical BP reducer',
            'Arjuna bark — strengthens heart muscle',
            'Ashwagandha — adaptogen, reduces cortisol',
            'Brahmi — calms nervous system',
            'Garlic (Lahsun) — dilates blood vessels',
        ],
        'remedy': (
            "Arjuna decoction: Boil 1 tsp Arjuna bark powder in 1 cup water for 10 min. "
            "Strain, add honey, drink warm twice daily. "
            "Also: eat 2 raw garlic cloves with warm water every morning."
        ),
        'diet': [
            'Eat: Bananas, watermelon, pomegranate, cucumber, garlic, oats',
            'Avoid: Salt (< 2 g/day), pickles, processed foods, caffeine, alcohol',
            'Increase: Potassium-rich foods (banana, sweet potato, spinach)',
            'Limit: Saturated fats, red meat',
        ],
        'yoga': [
            'Shavasana (Corpse pose) — 15 min with deep breathing',
            'Balasana (Child pose) — 5 min',
            'Viparita Karani (Legs up wall) — 10 min',
            'Pranayama: Chandra Bhedi (left-nostril breathing) 10 min',
            'Avoid: Hot yoga, inversions, intense cardio',
        ],
        'lifestyle': [
            'Check BP twice daily, log readings',
            'Reduce screen time, limit news consumption',
            'Practice progressive muscle relaxation before bed',
            'No smoking; limit alcohol strictly',
        ],
    },

    'Asthma': {
        'dosha': 'Vata',
        'herbs': [
            'Vasaka (Malabar nut) — bronchodilator, expels mucus',
            'Licorice root (Mulethi) — soothes airways',
            'Tulsi (Holy basil) — anti-allergic, strengthens lungs',
            'Pippali (Long pepper) — clears respiratory tract',
            'Ginger (Adrak) — reduces airway inflammation',
        ],
        'remedy': (
            "Steam inhalation: Add 5 drops eucalyptus oil + 1 tsp turmeric to boiling water. "
            "Inhale for 10 min, twice daily. "
            "Tulsi tea: Boil 10 fresh Tulsi leaves + ginger slice in 2 cups water, "
            "strain and drink warm 3× daily."
        ),
        'diet': [
            'Eat: Warm, cooked foods; ginger tea; turmeric milk; honey; figs',
            'Avoid: Cold foods/drinks, ice cream, banana, curd at night, fried snacks',
            'Keep: Regular meal times; do not eat late at night',
            'Warm water with honey + ginger every morning',
        ],
        'yoga': [
            'Pranayama (Breathing exercises) — 20 min daily (MOST important)',
            'Bhramari pranayama — 10 rounds, improves lung capacity',
            'Sukhasana with deep diaphragmatic breathing',
            'Setubandhasana (Bridge pose) — opens chest',
            'Avoid: Cold-weather outdoor exercise, intense cardio',
        ],
        'lifestyle': [
            'Keep home dust-free; use air purifier if possible',
            'Avoid strong perfumes, smoke, pet dander',
            'Keep inhaler accessible at all times',
            'Wear warm clothing in cold weather, cover nose/mouth',
        ],
    },

    'Thyroid Disorder': {
        'dosha': 'Kapha',
        'herbs': [
            'Kanchanar Guggul — classical thyroid herb',
            'Ashwagandha — regulates thyroid hormones',
            'Guggul — boosts T3/T4 conversion',
            'Brahmi — supports pituitary-thyroid axis',
            'Shankhpushpi — balances hormones naturally',
        ],
        'remedy': (
            "Kanchanar bark decoction: Boil 10 g bark in 400 ml water, "
            "reduce to 100 ml, drink on empty stomach. "
            "Also: mix 1 tsp Ashwagandha powder in warm milk before bed."
        ),
        'diet': [
            'Eat: Selenium-rich foods (Brazil nuts, sunflower seeds), iodine sources',
            'Hypothyroid: Warm cooked foods, light grains, leafy greens',
            'Hyperthyroid: Cooling foods (cucumber, coconut water, coriander)',
            'Avoid: Soy, raw cruciferous vegetables (broccoli, cauliflower) in excess',
        ],
        'yoga': [
            'Sarvangasana (Shoulder stand) — stimulates thyroid gland',
            'Matsyasana (Fish pose) — stretches throat/thyroid area',
            'Ujjayi pranayama — 10 min, activates throat chakra',
            'Halasana (Plow pose) — 5 breaths',
            'Daily 30 min moderate walk',
        ],
        'lifestyle': [
            'Sleep 7–8 hours; avoid late nights',
            'Take thyroid medication 30 min before breakfast',
            'Recheck TSH/T3/T4 every 3 months',
            'Reduce stress — it directly disrupts thyroid function',
        ],
    },

    'Anxiety Disorder': {
        'dosha': 'Vata',
        'herbs': [
            'Ashwagandha — reduces cortisol, calms nervous system',
            'Brahmi — improves cognitive function, reduces anxiety',
            'Shankhpushpi — promotes calmness and sleep',
            'Jatamansi — natural sedative, calms Vata',
            'Vacha (Calamus root) — improves focus and reduces panic',
        ],
        'remedy': (
            "Brahmi ghee: Add 1 tsp Brahmi powder to warm ghee, take at bedtime. "
            "Ashwagandha milk: 1 tsp Ashwagandha + 1 tsp honey in warm milk nightly. "
            "Warm sesame oil scalp massage (Shiroabhyanga) twice weekly."
        ),
        'diet': [
            'Eat: Warm, grounding foods — ghee, sesame, root vegetables, dates, almonds',
            'Avoid: Coffee, alcohol, processed sugars, energy drinks, cold raw foods',
            'Add: Warm milk with nutmeg at bedtime for better sleep',
            'Eat at fixed times — irregular eating worsens Vata',
        ],
        'yoga': [
            'Yoga Nidra (body scan meditation) — 20 min daily',
            'Shavasana — 15 min with guided breathing',
            'Slow Surya Namaskar — 5 rounds at gentle pace',
            'Nadi Shodhana (Alternate nostril breathing) — 15 min',
            'Evening walks in nature — 30 min barefoot on grass',
        ],
        'lifestyle': [
            'Maintain strict daily routine (Dinacharya) — Vata thrives on routine',
            'Limit news and social media to 30 min/day',
            'Journaling before bed — offload mental chatter',
            'Warm oil self-massage (Abhyanga) before shower daily',
        ],
    },

    'Obesity': {
        'dosha': 'Kapha',
        'herbs': [
            'Triphala — detoxifies gut, improves metabolism',
            'Guggul — breaks down fat tissue (Medha dhatu)',
            'Vrikshamla (Garcinia) — appetite suppressant',
            'Ginger — thermogenic, boosts fat burning',
            'Punarnava — removes water retention',
        ],
        'remedy': (
            "Morning fat-burning drink: 1 tsp honey + 1 tsp lemon + 1 tsp ginger juice "
            "in 1 glass warm water on empty stomach. "
            "Triphala churna: 1 tsp in warm water at bedtime for bowel health."
        ),
        'diet': [
            'Eat: Barley, millet, light vegetables, lentils, salads, warm soups',
            'Avoid: Dairy (except buttermilk), fried food, sweets, wheat excess, alcohol',
            'Practice: Intermittent fasting (16:8) or eat only 2 meals daily',
            'Chew each bite 20–30 times; eat in silence without screens',
        ],
        'yoga': [
            'Surya Namaskar — 12 rounds daily (builds heat)',
            'Navasana (Boat pose) — strengthens core',
            'Trikonasana — full-body engagement',
            'Kapalbhati pranayama — 10 min daily (metabolism booster)',
            'Cardio: 45 min brisk walk or cycling, 5 days/week',
        ],
        'lifestyle': [
            'Wake before sunrise; avoid daytime sleeping (increases Kapha)',
            'Use dry brushing (Garshana) before shower to stimulate lymph',
            'Keep a food diary; track weekly weight',
            'Eat largest meal at lunch, smallest at dinner before 7 pm',
        ],
    },

    'Anemia': {
        'dosha': 'Vata',
        'herbs': [
            'Lohasava — classical iron Ayurvedic tonic',
            'Draksha (Raisins) — natural iron source',
            'Punarnava — builds red blood cells',
            'Shatavari — supports blood building in women',
            'Amalaki (Amla) — vitamin C boosts iron absorption',
        ],
        'remedy': (
            "Iron boost: Soak 10 raisins + 1 fig in water overnight. "
            "Eat on empty stomach in the morning. "
            "Amla juice: 20 ml fresh amla juice with honey daily. "
            "Cook in iron cookware — leeches beneficial iron into food."
        ),
        'diet': [
            'Eat: Pomegranate, beetroot, dates, jaggery, spinach, lentils, liver (if non-veg)',
            'Pair: Vitamin C foods with iron (lemon on dal, amla juice)',
            'Avoid: Tea/coffee with meals (block iron absorption)',
            'Soak: Chickpeas, lentils, beans overnight before cooking',
        ],
        'yoga': [
            'Gentle Pranayama — Anulom Vilom 10 min daily',
            'Viparita Karani — boosts circulation',
            'Avoid: Intense exercise until hemoglobin improves',
            'Light Surya Namaskar — 5 gentle rounds',
            'Daily 20 min morning sunlight for Vitamin D',
        ],
        'lifestyle': [
            'Recheck hemoglobin every 4–6 weeks',
            'Cook in iron or cast-iron pan',
            'Rest adequately — do not push through fatigue',
            'Treat any parasitic infections (common cause in India)',
        ],
    },

    'GERD': {
        'dosha': 'Pitta',
        'herbs': [
            'Licorice root (Mulethi) — coats and heals stomach lining',
            'Shatavari — cooling, anti-ulcer',
            'Amla — reduces acidity, heals gastric mucosa',
            'Haritaki — regulates bowel, reduces reflux',
            'Coriander seeds (Dhaniya) — natural antacid',
        ],
        'remedy': (
            "Cooling drink: Soak 1 tsp coriander seeds in 1 cup water overnight. "
            "Strain and drink before meals. "
            "Mulethi chew: Chew a small piece of licorice root after meals. "
            "Amla candy or 20 ml amla juice before breakfast."
        ),
        'diet': [
            'Eat: Cucumber, coconut water, banana (ripe), cold milk, buttermilk, oats',
            'Avoid: Spicy food, tomatoes, citrus, coffee, alcohol, fried items, chocolate',
            'Never: Lie down within 3 hours of eating',
            'Eat: Small frequent meals; avoid skipping meals',
        ],
        'yoga': [
            'Vajrasana — sit for 10 min after every meal (only yoga safe after eating)',
            'Pawanmuktasana — relieves gas and bloating',
            'Avoid: Forward bends and inversions after meals',
            'Deep belly breathing — 5 min before meals',
            'Left-side sleeping reduces night reflux significantly',
        ],
        'lifestyle': [
            'Elevate head of bed by 6–8 inches',
            'Wear loose clothing — no tight belts',
            'Identify personal trigger foods; keep a food diary',
            'Do not eat < 3 hours before bed',
        ],
    },

    'Migraine': {
        'dosha': 'Pitta',
        'herbs': [
            'Butterbur (Petasites) — reduces migraine frequency',
            'Brahmi — calms overactive nervous system',
            'Sarpagandha — reduces headache intensity',
            'Shunthi (Dry ginger) — anti-inflammatory for head pain',
            'Nirgundi — analgesic for head and neck pain',
        ],
        'remedy': (
            "Peppermint oil: Dilute 2 drops in coconut oil, apply to temples + forehead. "
            "Ginger tea: Boil 1-inch fresh ginger in 2 cups water, add honey, sip slowly. "
            "Nasya (Nasal oil): 2 drops warm sesame oil in each nostril — classical Vata/Pitta remedy."
        ),
        'diet': [
            'Eat: Cooling foods — cucumber, mint, coconut, leafy greens, flaxseeds',
            'Avoid: Aged cheese, red wine, chocolate, MSG, artificial sweeteners, onion',
            'Stay hydrated: Dehydration is a major trigger',
            'Never skip meals — hypoglycemia triggers migraines',
        ],
        'yoga': [
            'Shavasana in dark, cool room during attack',
            'Jala Neti (nasal cleansing) — clears sinus pressure',
            'Sheetali pranayama (cooling breath) — 10 rounds daily',
            'Gentle neck stretches — release tension in cervical spine',
            'Avoid: Hot yoga, inverted poses during headache',
        ],
        'lifestyle': [
            'Maintain consistent sleep/wake time — even weekends',
            'Keep a migraine diary (triggers, duration, severity)',
            'Reduce screen time; use blue-light filter after sunset',
            'Apply cold pack to forehead, warm pack to neck during attacks',
        ],
    },

    'Arthritis': {
        'dosha': 'Vata',
        'herbs': [
            'Shallaki (Boswellia) — reduces joint inflammation',
            'Guggul — anti-inflammatory, rebuilds cartilage',
            'Ashwagandha — reduces pain and stiffness',
            'Nirgundi — topical and internal for joint pain',
            'Rasna — classical Ayurvedic joint herb',
        ],
        'remedy': (
            "Joint oil massage: Mix sesame oil + camphor, warm it, "
            "massage joints gently for 20 min daily before bath. "
            "Turmeric milk: 1 tsp turmeric + pinch black pepper in warm milk — nightly. "
            "Castor oil: 1 tsp at bedtime in warm water (reduces Vata in joints)."
        ),
        'diet': [
            'Eat: Warm, oily, nourishing foods — ghee, sesame, ginger, garlic, soups',
            'Avoid: Cold/raw foods, nightshades (tomato, potato, eggplant), curd, peas',
            'Add: Omega-3 sources (flaxseed, walnut, fish oil)',
            'Hydrate well: Warm water or herbal teas throughout day',
        ],
        'yoga': [
            'Gentle joint mobilisation exercises — morning rotation of all joints',
            'Balasana — relieves lower back and hip joint pressure',
            'Virabhadrasana (Warrior) — strengthens muscles around joints',
            'Water therapy: Exercise in warm water pool if available',
            'Avoid: High-impact exercises, running, jumping',
        ],
        'lifestyle': [
            'Apply warm compress to stiff joints every morning',
            'Maintain healthy weight — every kg reduces 4× load on knees',
            'Keep joints warm — avoid cold, damp environments',
            'Daily warm sesame oil self-massage before shower',
        ],
    },
}


# ── Fallback for unknown diseases ──────────────────────────
DEFAULT_RECOMMENDATION = {
    'dosha': 'Vata',
    'herbs': [
        'Ashwagandha — general adaptogen and immunity booster',
        'Triphala — detoxifier and digestive tonic',
        'Tulsi — immunity, stress, and respiratory support',
        'Turmeric — anti-inflammatory for all conditions',
        'Amla — richest natural Vitamin C source',
    ],
    'remedy': (
        "Golden milk: 1 tsp turmeric + 1/4 tsp black pepper + 1 tsp honey "
        "in 1 cup warm milk — drink nightly. "
        "Triphala water: 1 tsp Triphala in warm water at bedtime."
    ),
    'diet': [
        'Eat: Fresh whole foods, seasonal vegetables, lentils, warm soups',
        'Avoid: Processed food, excess sugar, alcohol, deep-fried items',
        'Maintain: Regular meal times, do not overeat',
        'Hydrate: Warm water or herbal teas throughout the day',
    ],
    'yoga': [
        'Surya Namaskar — 5–10 rounds daily',
        'Anulom Vilom pranayama — 10 min',
        'Meditation — 10 min daily for stress reduction',
        'Brisk walk — 30 min every morning',
    ],
    'lifestyle': [
        'Sleep by 10 pm, wake by 6 am',
        'Follow a consistent daily routine (Dinacharya)',
        'Practice gratitude and positive thinking',
        'Consult a qualified Ayurvedic practitioner for personalised treatment',
    ],
}


def get_recommendation(disease):
    """Return the full recommendation dict for a predicted disease."""
    return RECOMMENDATIONS.get(disease, DEFAULT_RECOMMENDATION)


# ── Dosha description cards ────────────────────────────────
DOSHA_INFO = {
    'Vata': {
        'elements': 'Air + Space',
        'qualities': 'Dry, light, cold, mobile, irregular',
        'personality': 'Creative, enthusiastic, quick-thinking but anxious when imbalanced',
        'imbalance_signs': 'Anxiety, insomnia, dry skin, constipation, joint pain, weight loss',
        'balance_tips': 'Routine, warmth, grounding, oil massage, warm cooked foods',
        'color': '#E8D5B7',
    },
    'Pitta': {
        'elements': 'Fire + Water',
        'qualities': 'Hot, sharp, oily, light, intense',
        'personality': 'Focused, ambitious, intelligent but irritable when imbalanced',
        'imbalance_signs': 'Inflammation, acidity, anger, skin rashes, excessive heat',
        'balance_tips': 'Cooling foods, moderation, avoiding overwork, nature exposure',
        'color': '#F4A460',
    },
    'Kapha': {
        'elements': 'Earth + Water',
        'qualities': 'Heavy, slow, cold, oily, stable, smooth',
        'personality': 'Calm, loyal, nurturing but sluggish when imbalanced',
        'imbalance_signs': 'Weight gain, mucus, lethargy, depression, slow digestion',
        'balance_tips': 'Stimulation, light foods, exercise, dry brushing, spices',
        'color': '#90EE90',
    },
}
