from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pickle
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# User-defined Core Aspects for specialized sarcasm mapping
CORE_ASPECT_KEYWORDS = ["app", "update", "feature", "camera", "battery", "performance", "price", "design"]

# Refined Aspect Extraction: Groups compound nouns (e.g., "battery life")
def extract_aspects(sentence):
    try:
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        aspects = []
        current_aspect = []
        for word, pos in tagged:
            # Check for noun tags
            if pos in ["NN", "NNS", "NNP", "NNPS"]:
                current_aspect.append(word.lower())
            else:
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []
        if current_aspect:
            aspects.append(" ".join(current_aspect))
            
        # Filter: Remove stop words, too short words, and duplicates
        stop_words = set(stopwords.words("english"))
        filtered_aspects = []
        for a in aspects:
            if a not in stop_words and len(a) > 2:
                filtered_aspects.append(a)
        
        return list(set(filtered_aspects))
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

# Load Deep Learning tools safely (handles broken torch DLLs on some Windows systems)
# NOTE: RoBERTa models require ~1GB RAM. On Render Free Tier (512MB), we skip them to avoid crash.
IS_LOW_MEM = os.environ.get("LOW_MEMORY_MODE", "false").lower() == "true"

bert_analyzer = None
sarcasm_analyzer = None

if not IS_LOW_MEM:
    try:
        from transformers import pipeline as hf_pipeline
        print("Transformers library loaded.")
        
        print("Initializing Dual RoBERTa Intelligence (Sentiment + Sarcasm)...")
        # 1. Sentiment Model
        SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        bert_analyzer = hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, device=-1) # Force CPU
        
        # 2. Sarcasm Detection Model
        SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-sarcasm"
        sarcasm_analyzer = hf_pipeline("sentiment-analysis", model=SARCASM_MODEL, tokenizer=SARCASM_MODEL, device=-1) # Force CPU
        
        print("Dual RoBERTa Models ready for deployment.")
    except Exception as e:
        print(f"Deep Learning load error or memory limit: {e}")
        bert_analyzer = None
        sarcasm_analyzer = None
else:
    print("LOW_MEMORY_MODE active. Skipping RoBERTa models to prevent crash.")
    bert_analyzer = None
    sarcasm_analyzer = None

# Load our high-performance ML pipeline (Logistic Regression)
MODEL_PATH = "sentiment_pipeline.pkl"
pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = pickle.load(open(MODEL_PATH, "rb"))
        print("ML Pipeline loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")

# Robust NLTK Data Management for Render/Cloud
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def ensure_nltk_resources():
    resources = [
        'punkt',
        'stopwords', 
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4',
        'punkt_tab',
        'averaged_perceptron_tagger_eng'
    ]
    for res in resources:
        try:
            print(f"Ensuring NLTK resource: {res}")
            nltk.download(res, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {res}: {e}")

ensure_nltk_resources()

# VADER for lexical sentiment analysis - ENHANCED for sarcasm
vader_analyzer = SentimentIntensityAnalyzer()

# CUSTOM VADER LEXICON: Upgrade with sarcastic, Hinglish, and backhanded triggers
vader_analyzer.lexicon.update({
    # Sarcastic/Ironic Slang (only if strongly indicative)
    'lipstick_on_a_pig': -2.5,
    'yeah_right': -2.0,
    
    # Hinglish & Cultural Sarcasm (Keep actually negative ones)
    'ghanta': -2.5, 'bakwas': -2.0, 'bekaar': -2.0, 'kachra': -2.2,
    'loot': -2.5, 'dhoka': -2.5, 'paisa_barbad': -2.5, 'dimag_kharab': -2.0,
    
    # Contextual Triggers (Actual negatives)
    'paperweight': -3.0, 'brick': -2.5, 'heater': -2.0, 'garbage': -2.5, 'trash': -2.5,
    'disaster': -3.0, 'joke': -2.2, 'useless': -2.5, 'broken': -2.0, 'slow': -1.5,
    
    # Positive Hinglish (Ensuring they are marked positive)
    'paisa_vasool': 2.5, 'fadu': 2.5, 'mast': 2.0, 'solid': 1.5, 'killer': 1.8,
    'zabardast': 2.0, 'kamaal': 2.0, 'wah': 1.5, 'shabash': 1.5, 'gajab': 2.0,
    'dhanyawad': 1.0, 'shukriya': 1.0, 'mubarak': 1.0, 'lajawab': 2.0
})

def get_word_sentiment(word):
    # Specialized word-level scorer for NLU
    v_raw = vader_analyzer.polarity_scores(word)['compound']
    t_raw = TextBlob(word).sentiment.polarity
    # Hybrid word score
    return round((0.7 * v_raw) + (0.3 * t_raw), 3)

def analyze_word_level(text):
    try:
        tokens = word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        word_analysis = []
        
        for word, pos in tagged:
            if len(word) < 2 and word not in ['a', 'i']: continue
            
            score = get_word_sentiment(word)
            label = "Neutral"
            if score > 0.1: label = "Positive"
            elif score < -0.1: label = "Negative"
            
            # Map POS tags to human readable roles
            roles = {
                'JJ': 'Description', 'JJR': 'Comparison', 'JJS': 'Superlative',
                'RB': 'Intensity', 'RBR': 'Comparison', 'RBS': 'Superlative',
                'NN': 'Object/Concept', 'NNS': 'Objects/Concepts', 
                'VB': 'Action', 'VBD': 'Past Action'
            }
            role = roles.get(pos[:2], 'Connector')
            
            word_analysis.append({
                "word": word,
                "score": score,
                "sentiment": label,
                "role": role,
                "pos": pos
            })
        return word_analysis
    except Exception as e:
        print(f"Word analysis error: {e}")
        return []

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    # Replace non-letters/numbers/punctuation with SPACE to avoid gluing words (like easier—highly)
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    # Keep important sentiment markers
    tokens = [w for w in tokens if w not in stop_words or w in ['not', 'no', 'never', 'but', 'however', 'very', 'highly']]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)


# Finds descriptive words (adjectives) near the aspect to explain "Why" the sentiment is such
def get_aspect_opinion(sentence, aspect):
    try:
        tokens = word_tokenize(sentence.lower())
        tagged = nltk.pos_tag(tokens)
        aspect_tokens = aspect.lower().split()
        
        # Find index of aspect
        idx = -1
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                idx = i
                break
        
        if idx == -1: return "not specified"
        
        # Search for adjectives (JJ) in a window around the aspect
        opinions = []
        start = max(0, idx - 3)
        end = min(len(tokens), idx + len(aspect_tokens) + 3)
        
        for i in range(start, end):
            word, pos = tagged[i]
            if pos.startswith("JJ") or pos in ["RB", "RBR", "RBS"]: # Adjectives or Adverbs
                opinions.append(word)
        
        return ", ".join(opinions) if opinions else "general context"
    except:
        return "general context"

def get_context_phrase(sentence, aspect):
    words = sentence.split()
    aspect_words = aspect.split()
    
    for i in range(len(words)):
        window = " ".join(words[i:i+len(aspect_words)]).lower()
        if aspect.lower() in window or window in aspect.lower():
            # Get a tighter context for specific aspect sentiment
            start = max(i - 3, 0)
            end = min(i + len(aspect_words) + 3, len(words))
            return " ".join(words[start:end])
    return sentence

def detect_sarcasm_expert(text, v_compound, subjectivity):
    text_lower = text.lower()
    cues = []
    tone = "sincere"
    
    # 1. Advanced Pattern Matchers
    # These are phrases that are ALMOST ALWAYS sarcastic or ironic
    sarcasm_triggers = [
        "yeah right", "lipstick on a pig", "glorified", "nice going", 
        "masterpiece (not)", "best app ever (sarcastic)"
    ]

    idiom_negatives = [
        "lipstick on a pig", "drop in the ocean", "waste of", 
        "head on a wall", "paperweight", "zero", "broken", "useless"
    ]

    hinglish_sarcasm = [
        "ghanta", "loot liya", "loot raha hai", "majak horha hai", "mazak hai",
        "paisa barbad", "dimag kharab", "bakwas software", "bekaar update",
        "heater ban gaya"
    ]

    # These words ARE positive but OFTEN used sarcastically ONLY if there is a negative contrast
    potential_sarcastic_positives = [
        "wonderful", "masterpiece", "pure magic", "fantastic", "awesome", 
        "excellent", "brilliant", "flawless", "great job", "so professional",
        "zabardast", "kamaal", "wah bhai", "gajab", "maza aa gaya"
    ]

    contrast_outcomes = [
        "battery lives next to a socket", "zero speed", "blur", "hang",
        "crash", "gayab", "slow", "broken", "kidney bechni", "open nahi",
        "dies in 2 hours", "dies in 1 hour", "not what i wanted",
        "security footage", "waiting", "seconds", "maza saste wala",
        "until they speak", "bright until they speak", "both be wrong",
        "unplug your life support", "ignore you", "one dollar", 
        "restart kar deta hai", "computer hang", "zero speed",
        "deleted all my data", "deleted my data", "brick", "paperweight",
        "restarts every 10 min", "stopped working", "garbage"
    ]

    # 2. Logic: Detection via Cultural & Contextual Irony
    
    # Check for Strong Ironic Triggers
    if any(p in text_lower for p in sarcasm_triggers):
        cues.append("ironic_trigger_phrase")
        tone = "sarcastic"

    # Check for Contrast (Positive claim vs Negative Outcome) - This is THE hallmark of sarcasm
    pos_claims = ["best", "great", "amazing", "love", "impressive", "clear", "fastest", "bright", "excellent", "superb"]
    neg_outcomes = ["worse", "slower", "crash", "0%", "zero", "blur", "hang", "broke", "heater", "until they speak", "useless"]
    
    # 1. Direct Contrast Check
    if any(p in text_lower for p in pos_claims + potential_sarcastic_positives) and any(n in text_lower for n in neg_outcomes + contrast_outcomes):
        cues.append("positive_claim_negative_outcome_contrast")
        tone = "ironic/sarcastic"

    # 2. Strong Negative Outcome with Positive start
    if any(c in text_lower for c in contrast_outcomes) and v_compound > 0.2:
        cues.append("ironic_negative_outcome")
        tone = "sarcastic"

    if any(h in text_lower for h in hinglish_sarcasm):
        cues.append("hinglish_irony")
        tone = "sarcastic (Hinglish)"

    # 3. VADER & TextBlob Synergy
    # If VADER is negative but TextBlob says it's emotional (subjective), it's likely sarcasm
    if v_compound < -0.1 and subjectivity > 0.6:
        cues.append("high_subjectivity_negative_context")
        if tone == "sincere": tone = "emotional_critique"

    is_sarcastic = len(cues) > 0
    return is_sarcastic, tone, cues

def analyze_sentiment_hybrid(text):
    text_lower = text.lower()
    clean = preprocess_text(text)
    
    # 1. Word-level NLU Analysis (Internal)
    word_nlu = analyze_word_level(text)
    
    # 2. VADER Score
    v_scores = vader_analyzer.polarity_scores(text)
    v_compound = v_scores['compound']
    
    # 3. TextBlob for Subjectivity & Polarity
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    blob_polarity = blob.sentiment.polarity
    
    # Penalize neutral-sounding emotional text (often hidden sarcasm)
    if subjectivity > 0.6 and abs(blob_polarity) < 0.2:
        blob_polarity -= 0.3 
    
    # 4. Deep Learning BERT Model (High Precision)
    bert_score = 0
    ml_label = None
    confidence = 0
    if bert_analyzer:
        try:
            res = bert_analyzer(text)[0]
            label = res['label'] 
            score = res['score']
            
            if label == 'LABEL_2' or label == 'POSITIVE':
                bert_score = score
                ml_label = "Positive"
            elif label == 'LABEL_0' or label == 'NEGATIVE':
                bert_score = -score
                ml_label = "Negative"
            else:
                bert_score = 0
                ml_label = "Neutral"
            confidence = float(score)
        except:
            bert_score = 0

    # Fallback/Synergy with ML Pipeline
    if pipeline:
        try:
            probs = pipeline.predict_proba([clean])[0]
            ml_res = pipeline.predict([clean])[0]
            confidence = max(confidence, float(max(probs)))
            if bert_score == 0:
                bert_score = [-1, 0, 1][ml_res]
                ml_label = ["Negative", "Neutral", "Positive"][ml_res]
        except:
            pass
    
    # 5. Sarcasm detection
    dl_sarcastic = False
    if sarcasm_analyzer:
        try:
            s_res = sarcasm_analyzer(text)[0]
            if s_res['label'] == 'LABEL_1':
                dl_sarcastic = True
                confidence = max(confidence, s_res['score'])
        except: pass

    is_sarcastic, tone, cues = detect_sarcasm_expert(text, v_compound, subjectivity)
    final_is_sarcastic = dl_sarcastic or is_sarcastic
    if dl_sarcastic: cues.append("transformers_sarcasm_detection")

    # 6. Hybrid Score Calculation
    # Synergy: If VADER and TextBlob both agree on strong positive, but ML says negative,
    # the ML model might be biased by the previous aggressive sarcasm labeling.
    
    # Check for custom Hinglish positive indicators in the text to help with trust
    hinglish_pos = ["zabardast", "kamaal", "gajab", "dhanyawad", "shukriya", "paisa vasool", "mast", "maza"]
    has_hinglish_pos = any(h in text_lower for h in hinglish_pos)
    
    if v_compound > 0.4 and (blob_polarity > 0.1 or has_hinglish_pos) and bert_score < 0:
        # Trust lexical analysis more when it's clearly positive (Lexical-ML Synergy)
        bert_score = 0.5 # Give it a slight positive push instead of just neutralizing
    
    base_score = (0.30 * v_compound) + (0.25 * blob_polarity) + (0.45 * bert_score)
    final_sentiment = "neutral"
    
    if final_is_sarcastic:
        final_sentiment = "Negative"
        tone = "sarcastic"
        base_score = min(base_score, -0.65)
    elif "extreme_negative_outcome" in cues:
        final_sentiment = "Negative"
        tone = "critical"
        base_score = min(base_score, -0.75)
    else:
        # Dynamic thresholding
        if base_score > 0.12: final_sentiment = "Positive"
        elif base_score < -0.12: final_sentiment = "Negative"
        else: final_sentiment = "Neutral"

    emoji_map = {"Positive": "Positive 😊", "Negative": "Negative 😡", "Neutral": "Neutral 😐", "Mixed": "Mixed 🧐"}
    display_sentiment = emoji_map.get(final_sentiment, final_sentiment)
    if final_is_sarcastic: display_sentiment += " (Sarcastic 😏)"

    return {
        "sentiment": final_sentiment.lower(),
        "display_sentiment": display_sentiment,
        "tone": tone,
        "sarcasm": final_is_sarcastic,
        "hidden_sentiment": "negative" if final_is_sarcastic else final_sentiment.lower(),
        "cues": cues,
        "confidence": confidence,
        "industry_score": round(base_score, 2),
        "word_analysis": word_nlu
    }

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>ToneIQ API Debugger</title>
            <style>
                body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #f4f4f9; }
                textarea { width: 100%; height: 100px; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; }
                button { background: #4a90e2; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
                button:hover { background: #357abd; }
                pre { background: #2d2d2d; color: #ccc; padding: 15px; border-radius: 4px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>🚀 ToneIQ API Test Page</h1>
            <p>Type any text below to test the backend analysis:</p>
            <form action="/api/analyze/text" method="get">
                <textarea name="text" placeholder="Enter text here (e.g., Wow, what a great feature! Not.)"></textarea>
                <br>
                <button type="submit">Analyze Sentiment</button>
            </form>
            <hr>
           <p><small>Endpoints: /api/analyze/text | /api/analyze/url</small></p>
        </body>
    </html>
    """

@app.route("/api/analyze/text", methods=["GET", "POST"])
def analyze_text():
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            sentence = data.get("text", "")
        else:
            sentence = request.args.get("text", "")
            
        if not sentence or not sentence.strip():
            return jsonify({"error": "Provide text"}), 400

        aspects = extract_aspects(sentence)
        results = {}

        # ALWAYS include Overall Sentiment Summary
        overall_res = analyze_sentiment_hybrid(sentence)
        results["Overall Summary"] = {
            "context": sentence,
            "sentiment": overall_res["display_sentiment"],
            "tone": overall_res["tone"],
            "is_sarcastic": overall_res["sarcasm"],
            "cues": overall_res["cues"],
            "confidence": f"{overall_res['confidence']:.2%}" if isinstance(overall_res["confidence"], float) else overall_res["confidence"],
            "industry_score": overall_res["industry_score"]
        }

        # Integrate User-Requested Aspect Sarcasm Logic
        text_lower = sentence.lower()
        for asp in CORE_ASPECT_KEYWORDS:
            if asp in text_lower:
                if overall_res["sarcasm"]:
                    results[f"Core Aspect: {asp.capitalize()}"] = {
                        "context": f"Found '{asp}' in sarcastic context",
                        "sentiment": "Negative 😡 (Sarcastic 😏)",
                        "opinion_words": "detected via sarcastic tone",
                        "tone": "sarcastic",
                        "is_sarcastic": True,
                        "confidence": "90% (Pattern Match)"
                    }
                elif asp not in aspects:
                    aspects.append(asp)

        # Add aspect-level analysis
        if aspects:
            noise_words = ["something", "anything", "everything", "someone", "anyone", "thing", "day", "time", "way", "lot"]
            for aspect in aspects:
                if aspect in noise_words: continue
                context = get_context_phrase(sentence, aspect)
                opinion = get_aspect_opinion(sentence, aspect)
                res = analyze_sentiment_hybrid(context)
                
                results[aspect] = {
                    "context": context,
                    "sentiment": res["display_sentiment"],
                    "opinion_words": opinion,
                    "tone": res["tone"],
                    "is_sarcastic": res["sarcasm"],
                    "confidence": f"{res['confidence']:.2%}" if isinstance(res["confidence"], float) else res["confidence"]
                }

        return jsonify({
            "status": "success",
            "sentence": sentence,
            "analysis": results
        })
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

import requests
from bs4 import BeautifulSoup

@app.route("/api/analyze/url", methods=["GET", "POST"])
def analyze_url():
    if request.method == "POST":
        data = request.get_json() or {}
        url = data.get("url", "")
    else:
        url = request.args.get("url", "")

    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Improved Scraping Logic
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. Extract Meta Data for better context
        title = soup.title.string if soup.title else "No Title"
        meta_desc = ""
        desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if desc_tag: meta_desc = desc_tag.get("content", "")

        # 2. Extract Body Text
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        
        # Focus on paragraphs and headings
        main_content = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3']):
            main_content.append(tag.get_text().strip())
        
        full_text = " ".join(main_content)
        full_text = re.sub(r"\s+", " ", full_text).strip()
        
        # Combine title, description and start of content for better summary
        analysis_input = f"{title}. {meta_desc}. {full_text[:3000]}"
        
        aspects = extract_aspects(analysis_input)
        results = {}

        # 3. Hybrid Analysis
        overall_res = analyze_sentiment_hybrid(analysis_input)
        results["Brand/Site Summary"] = {
            "title": title,
            "meta_description": meta_desc[:200] + "...",
            "sentiment": overall_res["display_sentiment"],
            "tone": overall_res["tone"],
            "is_sarcastic": overall_res["sarcasm"],
            "confidence": f"{overall_res['confidence']:.2%}" if isinstance(overall_res["confidence"], float) else overall_res["confidence"],
            "industry_score": overall_res["industry_score"]
        }

        # Add key aspects found in URL content
        if aspects:
            noise_words = ["something", "anything", "everything", "someone", "anyone", "thing", "day", "time", "way", "lot", "page", "home", "site"]
            count = 0
            for aspect in aspects:
                if count >= 10: break # Show top 10 aspects
                if aspect.lower() in noise_words or len(aspect) < 3: continue
                
                context = get_context_phrase(analysis_input, aspect)
                opinion = get_aspect_opinion(analysis_input, aspect)
                res = analyze_sentiment_hybrid(context)
                
                results[f"Aspect: {aspect.title()}"] = {
                    "sentiment": res["display_sentiment"],
                    "opinion": opinion,
                    "tone": res["tone"],
                    "is_sarcastic": res["sarcasm"]
                }
                count += 1
        
        return jsonify({
            "status": "success",
            "url": url,
            "metadata": {"title": title, "description": meta_desc},
            "analysis": results
        })
    except Exception as e:
        return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
