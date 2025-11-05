from typing import Any, Optional, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ActiveLoop, FollowupAction
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import torch
import requests
import logging
import json 
import os
from datetime import datetime
import re
import pandas as pd
import torch.nn.functional as F

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Path modelli addestrati
MODEL1_PATH = "./model1/best_model"
MODEL2_PATH = "./model2/best_model"

# inizializzazione modelli
tokenizer = RobertaTokenizer.from_pretrained(MODEL1_PATH)
model1 = RobertaForSequenceClassification.from_pretrained(MODEL1_PATH)
model2 = RobertaForSequenceClassification.from_pretrained(MODEL2_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device).eval()
model2.to(device).eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)


ner_tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
ner_model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


LOG_FOLDER = "session_logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

def make_safe_filename(s: str) -> str:
    safe = s.strip().lower()
    safe = safe.replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")
    if not safe:
        safe = "unknown"
    return safe[:30]


def save_full_session(tracker):
    name = tracker.get_slot("name") or ""
    age = tracker.get_slot("age") or ""
    occupation = tracker.get_slot("occupation") or ""
    interests = tracker.get_slot("interests") or ""

    if not tracker.get_slot("session_start_time"):
        tracker.slots["session_start_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_start = tracker.get_slot("session_start_time")

    user_signature = make_safe_filename(f"{name}_{age}_{session_start}")
    if not user_signature:
        user_signature = "unknown_user"

    session_file = os.path.join(LOG_FOLDER, f"{user_signature}.txt")
    metadata_file = os.path.join(LOG_FOLDER, f"{user_signature}_metadata.json")

    
    form_messages = []
    for event in tracker.events:
        if event.get("event") == "user":
            form_messages.append({"role": "user", "content": event.get("text", "")})
        elif event.get("event") == "bot":
            form_messages.append({"role": "bot", "content": event.get("text", "")})

    
    messages_log = tracker.get_slot("messages_log") or []

    
    seen_contents = set()
    all_messages = []

    def normalize_role(role):
        return "bot" if role in ["bot", "assistant", "system"] else "user"

    for msg in form_messages + messages_log:
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        role = normalize_role(msg.get("role", "user"))
        if text not in seen_contents:
            seen_contents.add(text)
            all_messages.append({"role": role, "content": text})


    with open(session_file, "w", encoding="utf-8") as f:
        for msg in all_messages:
            f.write(f"{msg['role'].upper()}: {msg['content']}\n")

    
    metadata = {
        "user_info": {"name": name, "age": age, "occupation": occupation, "interests": interests},
        "session_start": session_start,
        "last_update": datetime.now().isoformat(),
        "num_messages": len(all_messages),
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Sessione aggiornata per utente '{user_signature}' in {session_file}")


occupations = [
    # Students / Education
    "High school student", "University student", "Postgraduate student",
    "Teacher", "School teacher", "University professor", "Academic researcher",
    "Private tutor", "School counselor", "Librarian", "Education consultant",
    "Curriculum developer", "School principal", "Academic advisor", "Language teacher",
    "Early childhood educator", "Special education teacher", "Training specialist",
    
    # Healthcare
    "Doctor", "Dentist", "Nurse", "Pharmacist", "Psychologist", "Psychiatrist",
    "Therapist", "Medical assistant", "Paramedic", "Veterinarian",
    "Healthcare administrator", "Radiologist", "Anesthesiologist",
    "Occupational therapist", "Speech therapist", "Nutritionist", "Dietitian",
    "Dental hygienist", "Medical technologist", "Lab technician",
    "Caregiver", "Hospice worker", "Home health aide",

    # Generic work statuses
    "Worker", "Freelancer", "Self-employed", "Part-time worker", "Intern",
    "Unemployed", "Job seeker", "Homemaker", "Stay-at-home parent",
    "Retired", "Volunteer", "Gig worker", "Remote worker", "Digital nomad",

    # Tech & IT
    "Software developer", "Full stack developer", "Mobile developer", 
    "Game developer", "DevOps engineer", "Machine learning engineer",
    "AI researcher", "Data scientist", "Data analyst", "IT support specialist",
    "System administrator", "Cybersecurity analyst", "Cloud architect",
    "Blockchain developer", "Game designer", "QA tester", "Web designer",
    "UI/UX designer", "Database administrator", "IT auditor", "Tech blogger",

    # Engineering & Technical
    "Engineer", "Civil engineer", "Mechanical engineer", "Electrical engineer",
    "Industrial engineer", "Architect", "Construction worker", "Technician",
    "Mechanic", "Electrician", "Plumber", "Carpenter", "Welder", "Roofer",
    "HVAC technician", "Surveyor", "Glazier", "Mason",

    # Transport & Logistics
    "Truck driver", "Forklift operator", "Warehouse worker", "Logistics coordinator",
    "Supply chain manager", "Air traffic controller", "Pilot", "Flight attendant",
    "Ship captain", "Railway conductor", "Delivery driver", "Taxi driver",
    "Courier",

    # Art & Media
    "Artist", "Painter", "Illustrator", "Musician", "Composer", "Actor",
    "Filmmaker", "Photographer", "Video editor", "Graphic designer",
    "Fashion designer", "Interior designer", "Art director", "Animator",
    "Voice actor", "Model", "Creative director", "Comic artist", "Screenwriter",
    "Music producer", "DJ", "Tattoo artist",

    # Communication & Content
    "Journalist", "Writer", "Poet", "Content creator", "YouTuber",
    "Podcaster", "Influencer", "Social media manager",

    # Business & Management
    "Entrepreneur", "Business owner", "Startup founder", "Manager",
    "Project manager", "Product manager", "Salesperson", "Marketing specialist",
    "Financial analyst", "Accountant", "HR specialist", "Consultant",
    "Business analyst", "Recruiter", "Investment banker", "Trader",
    "Real estate agent", "Insurance agent", "Loan officer", "Auditor",
    "Economist", "Fundraiser", "Non-profit manager", "Executive assistant",
    "Office manager", "Administrative assistant", "Compliance officer",
    "Procurement specialist", "Operations manager", "Quality assurance specialist",
    "Event planner",

    # Legal
    "Lawyer", "Paralegal", "Judge", "Legal assistant", "Court clerk",
    "Legal advisor", "Mediator", "Notary public",

    # Public sector & Safety
    "Police officer", "Firefighter", "Military personnel", "Public servant",
    "Politician", "Social worker", "Community organizer", "NGO worker",
    "Immigration officer", "City planner", "Diplomat",

    # Science & Environment
    "Scientist", "Chemist", "Biologist", "Physicist", "Environmental scientist",
    "Geologist", "Lab researcher", "Clinical researcher", "Statistician",
    "Science communicator", "Ecologist", "Environmental consultant",
    "Agricultural engineer", "Forestry worker", "Park ranger", "Zookeeper",
    "Farmer", "Fisherman", "Beekeeper", "Landscape designer",

    # Service & Hospitality
    "Customer service agent", "Waiter", "Chef", "Cashier",
    "Retail worker", "Janitor", "Security guard", "Bartender",

    # Personal care & Lifestyle
    "Babysitter", "Pet sitter", "Dog walker", "Housekeeper", "Personal trainer",
    "Fitness coach", "Yoga instructor", "Life coach", "Motivational speaker",
    "Spiritual advisor", "Psychic",

    # Events & Entertainment
    "Magician", "Escort", "Club promoter", "Event host", "Auctioneer",

    # Neutral
    "No occupation", "Prefer not to say"
]

instruction_occ = "Represent the occupation mentioned in this sentence:"
occupation_embeddings = embedder.encode([[instruction_occ, occ] for occ in occupations], convert_to_tensor=True)

def classify_occupations_instructor(user_input, threshold=0.8, top_k=None):

    user_embedding = embedder.encode(
        [["What are the occupations of this person?:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, occupation_embeddings)[0]

    occupation_score_pairs = [
        (occupations[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    occupation_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        occupation_score_pairs = occupation_score_pairs[:top_k]

    return [label for label, score in occupation_score_pairs] if occupation_score_pairs else ["Other"]


# === Lista di interessi ===
interests = [
    "running", "jogging", "walking", "cycling", "swimming", "hiking", "climbing",
    "football", "soccer", "basketball", "tennis", "volleyball", "skiing", "snowboarding",
    "skating", "surfing", "martial arts", "boxing", "gym", "fitness", "yoga",
    "pilates", "aerobics", "dance fitness", "crossfit", "bodybuilding",
    "listening to music", "playing instruments", "singing", "composing music",
    "attending concerts", "music production", "DJing", "karaoke", "classical music",
    "rock music", "pop music", "jazz", "hip hop", "electronic music",
    "reading fiction", "reading non-fiction", "science fiction", "fantasy books",
    "mystery novels", "philosophy books", "self-help books", "poetry",
    "writing stories", "blogging", "journaling", "writing poetry", "creative writing",
    "video games", "mobile games", "MMORPGs", "strategy games", "board games",
    "card games", "chess", "Dungeons and Dragons", "puzzle games", "game development",
    "drawing", "painting", "sculpting", "digital art", "graphic design", "calligraphy",
    "photography", "film making", "video editing", "animation", "fashion design",
    "makeup art", "interior design", "crafting", "origami", "knitting", "sewing",
    "science", "physics", "astronomy", "biology", "chemistry", "mathematics", "philosophy",
    "psychology", "history", "politics", "geography", "languages", "learning new skills",
    "debating", "TED Talks", "documentaries", "museums", "archaeology",
    "cooking", "baking", "trying new recipes", "street food", "vegetarian food",
    "vegan cooking", "wine tasting", "coffee brewing", "craft beer",
    "traveling", "backpacking", "road trips", "exploring cities", "cultural exchange",
    "camping", "van life", "travel blogging", "airbnb experiences",
    "yoga", "meditation", "mindfulness", "journaling", "sleep optimization",
    "minimalism", "self-care", "productivity", "personal development",
    "gardening", "plants", "birdwatching", "fishing", "hunting", "camping",
    "forests", "mountains", "beaches", "animals", "pets", "dog walking",
    "volunteering at shelters", "horseback riding", "electronics",
    "DIY projects", "woodworking", "home improvement", "electronics repair",
    "model building", "mechanics", "robotics", "3D printing",
    "coding", "web development", "AI and machine learning", "tech news",
    "mobile apps", "gadget reviews", "cybersecurity", "hacking", "Linux",
    "open source", "startups", "digital marketing", "crypto", "NFTs",
    "volunteering", "activism", "environmental causes", "human rights",
    "religion", "spirituality", "astrology", "parenting", "family time",
    "socializing", "meeting new people", "clubbing", "networking", "nothing"
]

# === Macro categorie ===
macro_categories = {
    "Fitness & Sports": [
        "running", "jogging", "walking", "cycling", "swimming", "hiking", "climbing",
        "football", "soccer", "basketball", "tennis", "volleyball", "skiing", "snowboarding",
        "skating", "surfing", "martial arts", "boxing", "gym", "fitness", "yoga", "pilates",
        "aerobics", "dance fitness", "crossfit", "bodybuilding"
    ],
    "Music": [
        "listening to music", "playing instruments", "singing", "composing music",
        "attending concerts", "music production", "DJing", "karaoke", "classical music",
        "rock music", "pop music", "jazz", "hip hop", "electronic music"
    ],
    "Literature": [
        "reading fiction", "reading non-fiction", "science fiction", "fantasy books",
        "mystery novels", "philosophy books", "self-help books", "poetry",
        "writing stories", "blogging", "journaling", "writing poetry", "creative writing"
    ],
    "Gaming": [
        "video games", "mobile games", "MMORPGs", "strategy games", "board games",
        "card games", "chess", "Dungeons and Dragons", "puzzle games", "game development"
    ],
    "Arts": [
        "drawing", "painting", "sculpting", "digital art", "graphic design", "calligraphy",
        "photography", "film making", "video editing", "animation", "fashion design",
        "makeup art", "interior design", "crafting", "origami", "knitting", "sewing"
    ],
    "Science & Education": [
        "science", "physics", "astronomy", "biology", "chemistry", "mathematics", "philosophy",
        "psychology", "history", "politics", "geography", "languages", "learning new skills",
        "debating", "TED Talks", "documentaries", "museums", "archaeology"
    ],
    "Food & Cooking": [
        "cooking", "baking", "trying new recipes", "street food", "vegetarian food",
        "vegan cooking", "wine tasting", "coffee brewing", "craft beer"
    ],
    "Travel & Adventure": [
        "traveling", "backpacking", "road trips", "exploring cities", "cultural exchange",
        "camping", "van life", "travel blogging", "airbnb experiences"
    ],
    "Well-being & Lifestyle": [
        "yoga", "meditation", "mindfulness", "journaling", "sleep optimization",
        "minimalism", "self-care", "productivity", "personal development"
    ],
    "Nature & Outdoors": [
        "gardening", "plants", "birdwatching", "fishing", "hunting", "camping",
        "forests", "mountains", "beaches", "animals", "pets", "dog walking",
        "volunteering at shelters", "horseback riding"
    ],
    "Tech & Engineering": [
        "electronics", "DIY projects", "woodworking", "home improvement", "electronics repair",
        "model building", "mechanics", "robotics", "3D printing", 
        "coding", "web development", "AI and machine learning", "tech news",
        "mobile apps", "gadget reviews", "cybersecurity", "hacking", "Linux",
        "open source", "startups", "digital marketing", "crypto", "NFTs"
    ],
    "Social & Humanitarian": [
        "volunteering", "activism", "environmental causes", "human rights",
        "religion", "spirituality", "astrology", "parenting", "family time",
        "socializing", "meeting new people", "clubbing", "networking"
    ],
    "Other": ["nothing"]
}

# === Mappatura da interesse a macro-categoria ===
interest_to_macro = {
    interest: macro for macro, interest_list in macro_categories.items()
    for interest in interest_list
}

instruction_int = "Represent this interest category:"
interest_embeddings = embedder.encode([[instruction_int, int] for int in interests], convert_to_tensor=True)

def classify_interests_instructor(user_input, threshold=0.4, top_k=None):

    user_embedding = embedder.encode(
        [["What are the interests of this person?:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, interest_embeddings)[0]

    interest_score_pairs = [
        (interests[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    interest_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        interest_score_pairs = interest_score_pairs[:top_k]

    return [label for label, score in interest_score_pairs] if interest_score_pairs else ["Other"]

   
def classify_interests_with_macro(user_input, threshold=0.4, top_k=None):
    fine_labels = classify_interests_instructor(user_input, threshold=threshold, top_k=top_k)

    macro_set = set()
    for label in fine_labels:
        macro = interest_to_macro.get(label, "Other")
        macro_set.add(macro)

    return {
        "fine_labels": fine_labels,
        "macro_labels": list(macro_set)
    } 
    



common_health_labels_en = [
    "anxiety", "depression", "stress", "insomnia", "low self-esteem",
    "panic attacks", "burnout", "loneliness", "ocd", "ptsd",
    "drug addiction", "alcoholism", "smoking", "gambling addiction",
    "internet addiction", "social media addiction", "binge eating",
    "obesity", "anorexia", "poor nutrition", "lifestyle issues",
    "low motivation"
]

intruction_health = "Represent this health condition category:"
health_embeddings = embedder.encode([[intruction_health, label] for label in common_health_labels_en], convert_to_tensor=True)

def classify_health_condition_instructor(user_input, threshold=0.8, top_k=None):

    user_embedding = embedder.encode(
        [["Represent the health condition of this person:", user_input]], 
        convert_to_tensor=True
    )[0]

    cosine_scores = util.cos_sim(user_embedding, health_embeddings)[0]

    health_score_pairs = [
        (common_health_labels_en[i], float(score)) 
        for i, score in enumerate(cosine_scores) 
        if score >= threshold
    ]

    health_score_pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        health_score_pairs = health_score_pairs[:top_k]

    return [label for label, score in health_score_pairs] if health_score_pairs else ["Other"]


# richiesta modello LM Studio    
def get_model() -> str:
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        print(f"Error: {e}")
    return "mistral-7b-instruct-v0.3"


# ----- Azioni Personalizzate -----

class ValidateUserInfoForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_user_info_form"
    
    def validate_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        
        user_message = tracker.latest_message.get("text").strip()

        entities = ner_pipeline(user_message)
        name = None

        for ent in entities:
            if ent["entity_group"] == "PER":
                name = ent["word"]
                break


        if not name:
            if user_message.istitle() and " " not in user_message:

                name = user_message

        if name:
            dispatcher.utter_message(text=f"Nice to meet you, {name}!")
            return {"name": name}
        else:
            dispatcher.utter_message(text="Sorry, I couldn’t catch your name.")
            return {"name": None}
            
            
    def validate_age(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        try:
            age = int(slot_value)
            if 0 < age < 120:
                return {"age": age}
            else:
                dispatcher.utter_message(text="Age must be a number between 0 and 120.")
                return {"age": None}
        except Exception:
            dispatcher.utter_message(text="Age must be a number.")
            return {"age": None}

    def validate_occupation(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        user_message = tracker.latest_message.get("text").strip()
        return {"occupation": user_message}

    def validate_interests(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        user_message = tracker.latest_message.get("text").strip()
        # results = classify_interests_with_macro(slot_value, threshold=0.5, top_k=3)
        # macro_labels = results["macro_labels"]
        # fine_labels = results["fine_labels"]

        return {
            "interests": user_message,
            }


class ActionSubmitFormUserInfo(Action):

    def name(self) -> str:
        return "action_submit_user_info_form"

    def build_prompt(self, name, age, occupation, interests) -> str:
        return f"""
        You are an empathetic mental health support chatbot continuing an ongoing conversation with the user. 
        The user has just shared some information about themselves. You should respond naturally and seamlessly, 
        without any greetings or farewells.

        Here is what they've shared:
        - Name: {name}
        - Age: {age}
        - Occupation: {occupation}
        - Interests: {interests}

        Write a short, warm, and natural message (1–3 sentences) that:
        - Acknowledges and reflects something meaningful they’ve shared (e.g., age, job, hobbies, lifestyle)
        - Makes them feel heard, understood, and emotionally supported
        - Flows naturally as part of an ongoing chat (not as a new message)
        - Finish inviting them to share more about how they’re feeling today.

        IMPORTANT RULES:
        - Do NOT greet the user (no "Hi", "Hello", etc.)
        - Do NOT include their name unless it's natural mid-sentence
        - Do NOT include any closing phrases like "take care", "have a nice day", etc.
        - Do NOT ask “How are you?” or similar generic follow-ups
        - Do NOT talk about yourself
        - Do NOT restate your role or purpose
        - Keep the focus entirely on the user’s perspective

        Tone: friendly, non-judgmental, empathetic, and conversational."""

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        name = tracker.get_slot("name")
        age = tracker.get_slot("age")
        occupation = tracker.get_slot("occupation")
        interests = tracker.get_slot("interests")

        prompt = self.build_prompt(name, age, occupation, interests)

        model = get_model()
        headers = { "Content-Type": "application/json" }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content']
            else:
                reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."
        except Exception as e:
            print(f"Error: {e}")
            reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."

        dispatcher.utter_message(text=reply)
        

        
        return [
            SlotSet("form_completed", True),
            SlotSet("name", tracker.get_slot("name")),
            SlotSet("age", tracker.get_slot("age")),
            SlotSet("occupation", tracker.get_slot("occupation")),
            SlotSet("interests", tracker.get_slot("interests")),
            SlotSet("session_start_time", datetime.now().strftime("%Y%m%d_%H%M%S")),
            FollowupAction("action_start_interview")
        ]




class ActionStartInterview(Action):
    def name(self) -> str:
        return "action_start_interview"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [
            SlotSet("user_message", None),
            ActiveLoop(name="interview_form"),
            SlotSet("requested_slot", "user_message")
            ]



def analyze_profile(messages_log: str, current_profile: Dict[str, Any] = None) -> Dict[str, Any]:

    if current_profile is None:
        current_profile = {
            "mood": [],
            "personality_traits": [],
            "lifestyle": [],
            "social_and_relationships": [],
            "motivation": [],
            "thought_patterns": [],
            "possible_disorders": []
        }

    prompt = f"""
            Analyze the following conversation log and extract psychological indicators to update the following psychological profile:
            
            Current profile:
            {json.dumps(current_profile, indent=4)}
            
            Conversation so far:
            {messages_log}

            Instructions:
            - Reply ONLY with valid JSON (no explanations, no extra text).
            - Use lists for each field and include short descriptive phrases or keywords.
            - Base your analysis on the principles of Motivational Interviewing and Cognitive Behavioral Therapy where relevant.
            - If a category has no relevant information, return an empty list.

            {{
                "mood": [],                      # e.g., anxious, happy, sad ecc.
                "personality_traits": [],         # e.g., conscientious, introverted, perfectionist ecc.
                "lifestyle": [],                  # e.g., poor sleep, active, unhealthy diet ecc.
                "social_and_relationships": [],    # e.g., supportive friends, isolation, conflicts ecc.
                "motivation": [],                 # e.g., low motivation, high commitment ecc.
                "thought_patterns": [],           # e.g., negative self-talk, intrusive thoughts ecc.
                "possible_disorders": []          # e.g., anxiety, depression, OCD, stress-related disorder ecc.
            }}
            
            """

    try:
        model = get_model()
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "You are an assistant that ONLY replies with valid JSON.\n\n"
                            + prompt
                        )
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 200
            }
        )
        

        text = response.json()["choices"][0]["message"]["content"].strip()

        
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model response")
        
        json_data = json.loads(match.group())
        print(f"Extracted JSON: {json_data}")
        return json_data

    except Exception as e:
        print(f"Error parsing profile: {e}")
        return {}


question_clusters = {
    "emotional_tone": [
        "How have things been feeling for you lately?",
        "What’s your mood been like these past few days?",
        "Do you notice certain moments when you feel lighter or heavier?"
    ],
    "thoughts_selfview": [
        "When you think about yourself these days, what comes to mind?",
        "Do you ever find your thoughts going in circles?",
        "What kind of inner dialogue do you tend to have?"
    ],
    "energy_rest": [
        "How have your nights been?",
        "What’s your energy like throughout the day?"
    ],
    "relationships": [
        "How connected do you feel to people in your life right now?",
        "Who do you turn to when things are tough?"
    ],
    "work_balance": [
        "How does a typical day look for you lately?",
        "Is there anything in your routine that feels off balance?"
    ],
    "stress_changes": [
        "Has anything shifted recently that’s been on your mind?",
        "What’s been weighing on you, even quietly?"
    ],
    "coping_selfcare": [
        "When you need a bit of relief, what do you tend to do?",
        "Are there small things that bring you comfort?"
    ],
    "meaning_outlook": [
        "What’s been giving you a sense of purpose lately?",
        "When you look ahead, what do you hope feels different?"
    ],
    "change_continuity": [
        "When you think about yourself a few years ago, what feels different now?",
        "Are there things you used to enjoy that don’t feel the same anymore?"
    ]
}

topics = list(question_clusters.keys())
topic_embeddings = embedder.encode(topics, convert_to_tensor=True)

class ValidateInterviewForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_interview_form"

    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Text]:
        if tracker.get_slot("end_interview"):
            return []
        return ["user_message"]
    


    @staticmethod
    def detect_topic(user_message: str) -> str:
        user_emb = embedder.encode(user_message, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_emb, topic_embeddings)
        best_idx = int(cosine_scores.argmax())
        return topics[best_idx]

    def select_cluster(self, topic: str):
        return question_clusters.get(topic, question_clusters["emotional_tone"])

    @staticmethod
    def select_next_topic(user_message: str, recent_topics: List[str]) -> str:
        
        device = topic_embeddings.device if hasattr(topic_embeddings, "device") else torch.device("cpu")
        
        user_emb = embedder.encode(user_message, convert_to_tensor=True).to(device)
     
        cosine_scores = util.cos_sim(user_emb, topic_embeddings)[0]

        penalties = torch.ones(len(topics), device=device)

        for i, topic in enumerate(topics):
            if topic in recent_topics:
                penalties[i] = 0.5  

        penalized_scores = cosine_scores * penalties
        best_idx = int(penalized_scores.argmax())

        return topics[best_idx]

    
    def classify_message(self, user_input: str, tokenizer, model1, model2, device):
        MODEL1_MAP = {0: "neutral", 1: "non_neutral"}
        MODEL2_MAP = {0: "change", 1: "sustain"}

        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits1 = model1(**inputs).logits
            probs1 = F.softmax(logits1, dim=-1).cpu().numpy()[0]
        pred1 = int(probs1.argmax())
        label1 = MODEL1_MAP[pred1]
        conf1 = float(probs1[pred1])

        if label1 == "neutral":
            return {"label": "neutral", "confidence": conf1}

        with torch.no_grad():
            logits2 = model2(**inputs).logits
            probs2 = F.softmax(logits2, dim=-1).cpu().numpy()[0]
        pred2 = int(probs2.argmax())
        label2 = MODEL2_MAP[pred2]
        conf2 = float(probs2[pred2])

        return {"label": label2, "confidence": conf2}

    
    def user_wants_to_end(self, user_input: str, messages_log) -> bool:
        prompt = f"""
        You are an assistant analyzing a user message.
        User message: "{user_input}"

        Return True if the user clearly wants to end the conversation, no longer provide information or help, or if they express a desire to stop.

        Conversation so far:
        {json.dumps(messages_log, indent=4)}

        Examples of ending:
        - "Thank you."
        - "I think that's enough for now."
        - "No, I'm fine, thanks."
        - "That's all I wanted to share."
        - "I don't want to continue."
        - "Goodbye."

        Otherwise, return False. Respond only with True or False.
        """

        model = get_model()
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}

        try:
            response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content'].strip().lower()
                return reply == "true"
        except Exception as e:
            print(f"Error checking end intent: {e}")

        return False

    def enough_information(self, tracker: Tracker, user_input: str) -> bool:
        keys = [
            "mood",
            "personality_traits",
            "lifestyle",
            "social_and_relationships",
            "motivation",
            "thought_patterns",
            "possible_disorders"
        ]
        total_items = sum(len(tracker.get_slot(key) or []) for key in keys)
        return total_items >= 7

    
    def validate_user_message(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        try:
            if not isinstance(slot_value, str) or not slot_value.strip():
                dispatcher.utter_message(text="Please share something so I can understand you better.")
                return {"user_message": None}
            
            user_input = slot_value.strip()
            count = (tracker.get_slot("message_count") or 0) + 1
            talk_type = self.classify_message(user_input, tokenizer, model1, model2, device)

            messages_log = tracker.get_slot("messages_log") or []
            messages_log.append({"role": "user", "content": user_input, "talk_type": talk_type["label"], "confidence": talk_type["confidence"]})

            save_full_session(tracker)


            profile_data = analyze_profile(messages_log, {
                    "mood": tracker.get_slot("mood") or [],
                    "personality_traits": tracker.get_slot("personality_traits") or [],
                    "lifestyle": tracker.get_slot("lifestyle") or [],
                    "social_and_relationships": tracker.get_slot("social_and_relationships") or [],
                    "motivation": tracker.get_slot("motivation") or [],
                    "thought_patterns": tracker.get_slot("thought_patterns") or [],
                    "possible_disorders": tracker.get_slot("possible_disorders") or []
            })

            

            #Cambia prompt chiusura
            if (self.enough_information(tracker, user_input) and count >= 8):
                prompt = f"""
                    You are an empathetic mental health support chatbot.
                    The user has shared enough information, and your task is to respond with a short, kind, and empathetic closing message.
                    The last user message was: "{user_input}"
                    
                    Conversation so far: {json.dumps(messages_log, indent=4)}

                    Guidelines:
                    - Keep it brief (1-2 sentences).
                    - Be warm and appreciative of the user's openness.
                    - Avoid asking further questions or introducing new topics.
                """
                model = get_model()
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": model, 
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2, 
                    "max_tokens": 200
                    }

                try:
                    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
                    reply = response.json()['choices'][0]['message']['content'] if response.status_code == 200 else "Thank you for opening up today. I’m here whenever you’d like to talk again."
                except Exception as e:
                    print(f"Error: {e}")
                    reply = "Thank you for opening up today. I’m here whenever you’d like to talk again."

                dispatcher.utter_message(text=reply)
                messages_log.append({
                    "role": "assistant",
                    "content": reply
                })
                
                return {
                    "user_message": user_input,
                    "message_count": count,
                    "end_interview": True,
                    "messages_log": messages_log,
                    **{k: profile_data.get(k, []) for k in profile_data}
                }



            if count >= 4:
                prompt = f""" 
                    You are an empathetic mental health profiler chatbot continuing an ongoing conversation.
                    
                    Think about your chat with the person :{json.dumps(messages_log, indent=4)} 
                    And the profile you are building based on the conversation: {json.dumps(profile_data, indent=4)}
                    
                    Your goal is to generate a response that helps identify the user's potential mental health conditions,
                    asking only for information needed to confirm or rule them out.
                    Do not mention your goal or role in the conversation; ask questions naturally as part of the chat.

                    Avoid repeating questions already answered or topics already discussed.

                    The last message of the user is: "{user_input}"

                    Don't give advice, coping strategies, interpretations, or explanations. Focus only on understanding.
                """

                messages_for_model = [{"role": "user", "content": prompt}]

                model = get_model()
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": model, 
                    "messages": messages_for_model,
                    "temperature": 0.28, 
                    "max_tokens": 100,
                    "frequency_penalty": 0.8,
                    "presence_penalty": 0.6,
                    "stop": ["\n"] 
                    }

                try:
                    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
                    reply = response.json()['choices'][0]['message']['content'].strip().strip('"') if response.status_code == 200 else "Thank you for opening up today. I’m here whenever you’d like to talk again."
                except Exception as e:
                    print(f"Error: {e}")
                    reply = "Thank you for opening up today. I’m here whenever you’d like to talk again."

                messages_log.append({"role": "assistant", "content": reply})
                dispatcher.utter_message(text=reply)
                
                return {
                    "user_message": user_input,
                    "message_count": count,
                    "messages_log": messages_log,
                    "requested_slot": "user_message",
                    **{k: profile_data.get(k, []) for k in profile_data}
                }



            
            recent_topics = tracker.get_slot("recent_topics") or []
            topic = self.select_next_topic(user_input, recent_topics)
            cluster = self.select_cluster(topic)

            # rotazione degli ultimi 3 topic
            recent_topics.append(topic)
            if len(recent_topics) > 3:
                recent_topics.pop(0)

            
            name = tracker.get_slot("name")
            age = tracker.get_slot("age")
            occupation = tracker.get_slot("occupation")
            interests = tracker.get_slot("interests")

            prompt = f"""
            You are an empathetic mental health support chatbot continuing an ongoing conversation.

            Your goal is to explore the user's thoughts, feelings, and behaviors to gradually build a psychological profile.
            Do not give advice, coping strategies, interpretations, or explanations. Focus only on understanding and reflection.

            User context:
            - Name: {name}
            - Age: {age}
            - Occupation: {occupation}
            - Interests: {interests}

            Profile summary:
            {json.dumps(profile_data, indent=4)}

            Current topic: {topic}
            Potential inspiration questions: {cluster}

            Guidelines:
            - Ask only **one open-ended question** at a time.
            - Keep the question **short, 1 sentence, under 30 words**.
            - Be **empathetic, natural, and conversational** but not intrusive.S
            - **Do not** include reasons, explanations, or follow-ups like “this might help” or “we could explore this together”.
            - **Do not** give advice, suggestions, or solutions.
            - **Do not** summarize or repeat context; just continue the conversation naturally.
            """

            messages_for_model = [{"role": "user", "content": prompt}]
            messages_for_model.extend([{"role": msg["role"], "content": msg["content"]} for msg in messages_log])

            model = get_model()
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": messages_for_model,
                "temperature": 0.35,
                "max_tokens": 100,
                "frequency_penalty": 0.8,
                "presence_penalty": 0.6,
                "stop": ["\n"] 
            }

            try:
                response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
                reply = response.json()['choices'][0]['message']['content'] if response.status_code == 200 else "Would you like to share a bit more about how things have been feeling for you lately?"
            except Exception as e:
                print(f"Error: {e}")
                reply = "Would you like to share a bit more about how things have been feeling for you lately?"

            messages_log.append({"role": "assistant", "content": reply})
            dispatcher.utter_message(text=reply)

            return {
                "user_message": user_input,
                "message_count": count,
                "messages_log": messages_log,
                "recent_topics": recent_topics,
                "requested_slot": "user_message",
                **{k: profile_data.get(k, []) for k in profile_data}
            }

        except Exception as e:
            dispatcher.utter_message(text=f"Errore interno: {e}")
            return {"user_message": None}



class ActionSubmitInterviewForm(Action):
    def name(self) -> str:
        return "action_submit_interview_form"

    def build_prompt(self, mood, personality_traits, lifestyle, social_and_relationships, motivation, thought_patterns,possible_disorders, messages_log) -> str:
            return f"""            
            You are a professional psychologist and behavioral scientist. 
            
            This is a conversation between a person and a chatbot: {messages_log}
            This is the psychological profile inferred through a machine learning–based analysis of the user’s conversation with the chatbot:
                - Mood: {mood}
                - Personality Traits: {personality_traits}
                - Lifestyle: {lifestyle}
                - Social and Relationships: {social_and_relationships}
                - Motivation: {motivation}
                - Thought Patterns: {thought_patterns}
                - Possible Disorders: {possible_disorders}

             
               Which kind of psychological profile of the person emerges from the conversation? 
               Is he possibly experiencing any of those conditions?
                    - Anxiety disorders
                    - Depressive disorders
                    - Post-traumatic stress disorder (PTSD)
                    - Obsessive-compulsive disorder (OCD)
                    - Bipolar disorder
                    - Schizophrenia and related psychotic disorders
                    - Attention-deficit/hyperactivity disorder (ADHD)
                    - Eating disorders
                    - Substance use disorders
                    - Personality disorders
            """
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        

        mood = tracker.get_slot("mood") or [],
        personality_traits = tracker.get_slot("personality_traits") or [],
        lifestyle = tracker.get_slot("lifestyle") or [],
        social_and_relationships = tracker.get_slot("social_and_relationships") or [],
        motivation = tracker.get_slot("motivation") or [],
        thought_patterns = tracker.get_slot("thought_patterns") or [],
        possible_disorders = tracker.get_slot("possible_disorders") or [],
        messages_log = tracker.get_slot("messages_log") or []
        
        prompt = self.build_prompt(mood, personality_traits, lifestyle, social_and_relationships, motivation, thought_patterns, possible_disorders, messages_log)

        model = get_model()
        headers = { "Content-Type": "application/json" }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 300
            }
        

        try:
            response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content']
            else:
                reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."
        except Exception as e:
            print(f"Error: {e}")
            reply = "Thanks for completing the form! If you’d like, feel free to tell me if there’s anything you’d like to work on or explore together."

        dispatcher.utter_message(text=reply)

        messages_log.append({
            "role": "assistant",
            "content": reply
        })
        
        profile_data = {
            "mood": mood,
            "personality_traits": personality_traits,
            "lifestyle": lifestyle,
            "social_and_relationships": social_and_relationships,
            "motivation": motivation,
            "thought_patterns": thought_patterns,
            "possible_disorders": possible_disorders,
        }
        
        profile_str = "PROFILE_DATA: " + json.dumps(profile_data, ensure_ascii=False, indent=2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile_message_content = f"{profile_str}\n_Saved at: {timestamp}_"

        messages_log.append({
            "role": "assistant",
            "content": profile_message_content
        })

       
        save_full_session(tracker)



        return []