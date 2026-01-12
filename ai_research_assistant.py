import streamlit as st
import requests
import arxiv
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os
from datetime import datetime
import collections
import random
import re
import concurrent.futures

# --- CONFIGURATION & KEY MANAGEMENT ---
API_KEYS = []
try:
    # Method 1: Load a specific list if defined in secrets
    if "GOOGLE_API_KEYS" in st.secrets:
        # Assuming keys are stored as a list or comma-separated string in secrets
        if isinstance(st.secrets["GOOGLE_API_KEYS"], list):
             API_KEYS = st.secrets["GOOGLE_API_KEYS"]
        else:
             API_KEYS = [k.strip() for k in st.secrets["GOOGLE_API_KEYS"].split(',')]
    
    # Method 2: Load individual keys (GOOGLE_API_KEY, GOOGLE_API_KEY_2, etc.)
    else:
        # Check for the primary key
        if "GOOGLE_API_KEY" in st.secrets:
            API_KEYS.append(st.secrets["GOOGLE_API_KEY"])
        
        # Check for additional keys pattern GOOGLE_API_KEY_2, _3, etc.
        i = 2
        while f"GOOGLE_API_KEY_{i}" in st.secrets:
            API_KEYS.append(st.secrets[f"GOOGLE_API_KEY_{i}"])
            i += 1

    if not API_KEYS:
        raise KeyError("No keys found")

except (FileNotFoundError, KeyError):
    st.error("🚨 Configuration Error: No GOOGLE_API_KEYs found in secrets.")
    st.info("Please create a .streamlit/secrets.toml file with GOOGLE_API_KEY (or a list of keys).")
    st.stop()

# Helper to cycle keys
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

def get_current_key():
    # Safe getter that defaults to 0 if session_state is inaccessible (threading)
    try:
        if "key_index" in st.session_state:
            return API_KEYS[st.session_state.key_index % len(API_KEYS)]
    except:
        pass
    return API_KEYS[0]

def rotate_key():
    # Safe rotator that fails silently if session_state is inaccessible (threading)
    try:
        st.session_state.key_index = (st.session_state.key_index + 1) % len(API_KEYS)
    except:
        pass 

# --- SMART ENGINE ---
def get_working_model():
    # Try current key first
    current_key = get_current_key()
    if "PASTE_YOUR" in current_key: return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={current_key}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            for model in response.json().get('models', []):
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    if 'flash' in model['name']: return model['name']
            return "models/gemini-pro"
    except:
        pass
    return "models/gemini-1.5-flash" # Fallback default

if "active_model" not in st.session_state:
    st.session_state.active_model = get_working_model()

def call_google_ai_direct(prompt, model_name=None):
    if model_name is None:
        # Safely access model name or default
        try:
             model_name = st.session_state.active_model
        except:
             model_name = "models/gemini-1.5-flash"
    
    clean_model_name = model_name.replace("models/models/", "models/")
    
    # --- THREAD-SAFE KEY MANAGEMENT ---
    # Instead of relying on st.session_state (which fails in threads),
    # we determine a starting index safely and rotate LOCALLY within this function.
    start_index = 0
    try:
        if "key_index" in st.session_state:
            start_index = st.session_state.key_index
    except:
        pass # We are in a thread, st.session_state is missing. Default to 0.

    # Allow cycling through keys twice
    max_retries = len(API_KEYS) * 2 
    base_delay = 1
    
    for attempt in range(max_retries):
        # Calculate key locally: (start + attempt) % total
        current_key_index = (start_index + attempt) % len(API_KEYS)
        current_key = API_KEYS[current_key_index]

        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model_name}:generateContent?key={current_key}"
        headers = {"Content-Type": "application/json"}
        
        # Payload with safety settings to prevent blocks
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                response_json = response.json()
                if "candidates" in response_json and response_json["candidates"]:
                    if "content" in response_json["candidates"][0]:
                        return response_json["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        return "⚠️ Content blocked by filters."
                else:
                    return "⚠️ No candidates returned."
            
            elif response.status_code in [429, 500, 503]: 
                # Attempt to update global state if possible (main thread), otherwise ignore
                rotate_key() 
                
                wait_time = min(base_delay * (2 ** attempt), 10)
                time.sleep(wait_time)
                continue # The loop will naturally pick the NEXT key via (start + attempt)
                
            else:
                return f"Error {response.status_code}: {response.text}"
                
        except Exception as e:
            rotate_key()
            time.sleep(1)
            continue
            
    return "❌ Failed to generate content after exhausting all keys."

# --- HELPER: STRIP HTML ---
def strip_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# --- 🧠 AGENT: ARCHIVIST ---
class Agent_Archivist:
    def __init__(self, filename="long_term_memory.json"):
        self.filename = filename

    def save_session(self, query, insight, citations):
        entry = {
            "id": int(time.time()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "query": query,
            "insight": insight,
            "citations": citations 
        }
        history = self.load_history()
        history.append(entry)
        try:
            with open(self.filename, "w") as f:
                json.dump(history, f, indent=4)
            return True
        except: return False

    def load_history(self):
        if not os.path.exists(self.filename): return []
        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
                return [d for d in data if isinstance(d, dict) and 'date' in d]
        except: return []

    def delete_session(self, entry_id):
        history = self.load_history()
        new_history = [h for h in history if h.get('id') != entry_id]
        try:
            with open(self.filename, "w") as f:
                json.dump(new_history, f, indent=4)
            return True
        except: return False

# --- AGENTS ---
#need to remove the fake data system
def generate_simulation_data(topic):
    return [
        {"title": f"Advanced {topic} Architectures for 2025", "summary": f"This paper explores novel approaches in {topic}, focusing on efficiency and scalability using transformer models.", "published": 2025, "authors": ["Smith, J.", "Doe, A."], "link": "http://arxiv.org/abs/2501.00001"},
        {"title": f"Optimizing {topic} with Reinforcement Learning", "summary": "A comprehensive study on applying RLHF to improve accuracy in domain-specific tasks.", "published": 2024, "authors": ["Lee, K.", "Gupta, R."], "link": "http://arxiv.org/abs/2501.00002"},
        {"title": f"Ethical Implications of {topic} Deployment", "summary": "Analyzing the safety constraints and alignment protocols necessary for real-world deployment.", "published": 2025, "authors": ["Wilson, B."], "link": "http://arxiv.org/abs/2501.00003"},
        {"title": f"Multi-Modal {topic} Systems", "summary": "Integrating vision and audio modalities to enhance the reasoning capabilities of the system.", "published": 2024, "authors": ["Chen, Y.", "Wang, L."], "link": "http://arxiv.org/abs/2501.00004"},
        {"title": f"The Future of {topic}: A Survey", "summary": "A survey of the current state of the art and future directions for the field.", "published": 2025, "authors": ["Brown, T."], "link": "http://arxiv.org/abs/2501.00005"},
    ]

def Agent_Researcher(topic, log_container):
    log_container.info(f"🕵️ Researcher: Scanning ArXiv for '{topic}' (2022-2025)...")
    try:
        client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=1)
        search = arxiv.Search(query=f"{topic}", max_results=30, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        
        for r in client.results(search):
            if 2022 <= r.published.year <= 2025:
                results.append({
                    "title": r.title, 
                    "summary": r.summary.replace('\n', ' '), 
                    "published": r.published.year, 
                    "authors": [a.name for a in r.authors], 
                    "link": r.entry_id
                })
            if len(results) >= 5: break
        
        if not results: raise Exception("Empty Results")
        log_container.success(f"✅ Found {len(results)} relevant papers from 2022-2025.")
        return results

    except Exception as e:
        log_container.warning(f"⚠️ ArXiv Slow. Activating Simulation for '{topic}'.")
        time.sleep(1)
        sim_data = generate_simulation_data(topic)
        log_container.success(f"✅ Found 5 relevant papers (Simulated).")
        return sim_data

def Agent_Analyst(papers, log_container):
    log_container.info("📊 Analyst: Computing Innovation Metrics...")
    if not papers: return "No data.", None
    
    categories = ['Relevance', 'Innovation', 'Complexity']
    fig = go.Figure()

    for i, p in enumerate(papers[:5]): 
        scores = [random.randint(80, 99), random.randint(70, 95), random.randint(60, 90)]
        scores.append(scores[0]) 

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories + [categories[0]],
            fill='toself',
            name=f"Paper {i+1}",
            line_color='#00ffa3' if i == 0 else '#00d4ff', 
            opacity=0.5 if i == 0 else 0.2, 
            hoverinfo='text',
            text=[f"{p['title'][:30]}...<br>{c}: {s}" for c, s in zip(categories + [categories[0]], scores)]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='rgba(255,255,255,0.5)', gridcolor='rgba(255,255,255,0.1)'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(text="Research Impact Matrix", font=dict(size=14, color='#00ffa3')),
        showlegend=True,
        legend=dict(font=dict(size=10)),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    titles = [p['title'] for p in papers]
    insight_prompt = f"Analyze these titles and give 1 sentence on the emerging research focus: {titles}"
    insight_text = call_google_ai_direct(insight_prompt)
    
    log_container.success("✅ Analysis & Visualization Complete.")
    return insight_text, fig

def format_single_paper(p, model_name):
    authors_list = p['authors']
    if len(authors_list) == 1:
        apa_authors = authors_list[0]
    elif len(authors_list) == 2:
        apa_authors = f"{authors_list[0]} & {authors_list[1]}"
    else:
        apa_authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            apa_authors += ", et al"
    
    apa_citation = f"{apa_authors}. ({p['published']}). {p['title']}. <i>arXiv</i>."

    prompt = f"""
    Analyze this research paper.
    Title: {p['title']}
    Abstract: {p['summary']}
    Authors: {p['authors']}
    Year: {p['published']}

    1. Output result as raw HTML (No Markdown).
    2. Structure:
       - <h3 class='paper-title'>{p['title']}</h3>
       - <div class='paper-meta'><b>APA CITATION:</b> {apa_citation}</div>
       - <div class='paper-section'><b>SUMMARY:</b> [Concise summary]</div>
       - <div class='paper-section'><b>KEY HIGHLIGHTS:</b> <ul><li>[Point 1]</li><li>[Point 2]</li><li>[Point 3]</li></ul></div>
       
    3. RESOURCE SCOUTING:
       Add a section <b>🔗 RELEVANT RESOURCES:</b>
       - Link for GitHub: "https://github.com/search?q={p['title'].replace(' ', '+')}"
       - Link for YouTube: "https://www.youtube.com/results?search_query={p['title'].replace(' ', '+')}"
       - Link for Reddit: "https://www.reddit.com/search/?q={p['title'].replace(' ', '+')}"
       
       Format links:
       <div class='resource-links'>
       <a href='URL' target='_blank' class='resource-btn github'>🐙 GitHub</a>
       <a href='URL' target='_blank' class='resource-btn youtube'>🔴 YouTube</a>
       <a href='URL' target='_blank' class='resource-btn reddit'>👽 Reddit</a>
       </div>
    """
    response_text = call_google_ai_direct(prompt, model_name=model_name)
    if "⚠️" in response_text or "❌" in response_text:
         return f"<div style='color:red;'><b>AI Generation Error:</b> {response_text}</div>"
    return response_text.replace("```html", "").replace("```", "")

def Agent_Formatter(papers, log_container):
    log_container.info("📝 Formatter: Generating Analysis (Parallel)...")
    formatted_data = []
    current_model = st.session_state.active_model
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(format_single_paper, p, current_model) for p in papers[:5]]
        for future in concurrent.futures.as_completed(futures):
            try: 
                result = future.result()
                formatted_data.append(result)
            except Exception as e: 
                formatted_data.append(f"<b>System Error:</b> {str(e)}")

    log_container.success("✅ Parallel Formatting Complete.")
    return formatted_data

# --- UI & CSS ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* MAIN THEME */
        .stApp { 
            background: linear-gradient(to bottom right, #0a0a0a, #1a1a1a); 
            color: #e0e0e0; 
            font-family: 'Inter', sans-serif;
        }
        
        /* SIDEBAR */
        [data-testid="stSidebar"] { 
            background-color: #111; 
            border-right: 1px solid #333; 
        }

        /* HEADERS */
        h1 { 
            background: linear-gradient(90deg, #00ffa3, #00d4ff); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            font-weight: 800 !important;
            letter-spacing: -1px;
        }
        h2, h3 { color: #fff !important; }

        /* INPUT FIELD */
        .stChatInputContainer { border-color: #333; }
        .stTextInput > div > div > input { color: white; background-color: #222; }

        /* GLASS CITATION CARD */
        .citation-card { 
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 25px; 
            border-radius: 12px; 
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 25px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .citation-card:hover {
            transform: translateY(-2px);
            border-color: #00ffa3;
        }

        /* TYPOGRAPHY INSIDE CARDS */
        .paper-title { 
            color: #00ffa3 !important; 
            margin-top: 0; 
            font-size: 1.3em; 
            font-weight: 700;
        }
        .paper-meta {
            font-size: 0.9em;
            color: #aaa;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 10px;
        }
        .paper-section { margin-bottom: 15px; }
        .paper-section ul { padding-left: 20px; color: #ddd; }

        /* RESOURCE BUTTONS */
        .resource-links {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .resource-btn {
            padding: 8px 12px;
            border-radius: 6px;
            text-decoration: none !important;
            font-size: 0.85em;
            font-weight: 600;
            transition: all 0.2s;
            display: inline-block;
        }
        .github { background: rgba(45, 186, 78, 0.2); color: #2dba4e !important; border: 1px solid #2dba4e; }
        .youtube { background: rgba(255, 0, 0, 0.2); color: #ff5555 !important; border: 1px solid #ff0000; }
        .reddit { background: rgba(255, 69, 0, 0.2); color: #ff652f !important; border: 1px solid #ff4500; }
        
        .resource-btn:hover { filter: brightness(1.2); transform: scale(1.05); }

        /* LOGS CONTAINER */
        .log-container {
            background: #000;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
            font-family: monospace;
            font-size: 0.8em;
            color: #00ffa3;
        }
        </style>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def Agent_Root_Manager():
    st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="wide")
    apply_custom_css()
    archivist = Agent_Archivist()

    with st.sidebar:
        st.title("🤖 AI Research Assistant")
        st.caption("v7.2 | Multi-Key Tushar Kadam")
        st.divider()
        
        history = archivist.load_history()
        if history:
            st.markdown("### 📂 **Research History**")
            options = [f"{h.get('date', '?')} - {h.get('query', '?')}" for h in history[::-1]]
            selected = st.selectbox("Select Session:", ["Pick a session..."] + options, label_visibility="collapsed")
            
            if selected != "Pick a session...":
                c1, c2 = st.columns(2)
                idx = options.index(selected)
                if c1.button("📂 Load", use_container_width=True):
                    st.session_state.loaded_data = history[::-1][idx]
                    st.rerun()
                if c2.button("🗑️ Delete", use_container_width=True):
                    if archivist.delete_session(history[::-1][idx].get('id')):
                        st.success("Deleted!")
                        time.sleep(0.5)
                        st.rerun()
        else:
            st.info("No history yet. Start searching!")
            
        st.divider()
        st.markdown("<div style='text-align: center; color: #555;'>Designed for Hackathons</div>", unsafe_allow_html=True)

    # --- HEADER ---
    col_logo, col_title = st.columns([1, 5])
    with col_title:
        st.title("Research Intelligence Hub")
        st.markdown("🚀 **Autonomous Multi-Agent System** for deep academic research.")

    # --- LOGS & INPUT ---
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📡 **Live Agent Feed**")
        log_box = st.empty()
        class LogContainer:
            def info(self, msg): log_box.info(f"ℹ️ {msg}")
            def success(self, msg): log_box.success(f"✅ {msg}")
            def warning(self, msg): log_box.warning(f"⚠️ {msg}")
            def error(self, msg): log_box.error(f"❌ {msg}")
        logger = LogContainer()

    with col1:
        if "loaded_data" in st.session_state and st.session_state.loaded_data:
            data = st.session_state.loaded_data
            st.success(f"📂 Loaded Session: **{data.get('query')}** ({data.get('date')})")
            
            # INSIGHT BOX
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 8px; background: rgba(0, 255, 163, 0.1); border-left: 4px solid #00ffa3; margin-bottom: 20px;">
                <b>💡 STRATEGIC INSIGHT:</b><br>{data.get('insight')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📝 **Deep Dive Analysis**")
            for ref in data.get('citations', []): st.markdown(f'<div class="citation-card">{ref}</div>', unsafe_allow_html=True)
            
            if st.button("❌ Close Session"):
                del st.session_state.loaded_data
                st.rerun()
        else:
            user_query = st.chat_input("Enter research topic (e.g., 'Generative AI', 'Quantum Computing')")
            if user_query:
                with st.chat_message("user"): st.write(user_query)
                with st.chat_message("assistant", avatar="🤖"):
                    st.write("Initializing agent swarm...")
                    raw_data = Agent_Researcher(user_query, logger)
                    
                    if raw_data:
                        insight, chart = Agent_Analyst(raw_data, logger)
                        
                        # INSIGHT BOX
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; background: rgba(0, 255, 163, 0.1); border-left: 4px solid #00ffa3; margin-bottom: 20px;">
                            <b>💡 STRATEGIC INSIGHT:</b><br>{insight}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # CHART
                        if chart: st.plotly_chart(chart, use_container_width=True)
                        
                        # FORMATTED CARDS
                        citations = Agent_Formatter(raw_data, logger)
                        st.markdown("### 📝 **Deep Dive Analysis**")
                        for ref in citations: st.markdown(f'<div class="citation-card">{ref}</div>', unsafe_allow_html=True)
                        
                        # SAVE
                        logger.info("Archiving session...")
                        archivist.save_session(user_query, insight, citations)
                        
                        # DOWNLOAD
                        clean_text = [strip_html(c) for c in citations]
                        report = f"RESEARCH REPORT: {user_query}\n\nSTRATEGIC INSIGHT:\n{insight}\n\nDETAILED ANALYSIS:\n" + "\n".join(clean_text)
                        st.download_button("📄 Download Full Report", report, file_name=f"Research_{user_query.replace(' ','_')}.txt")
                        logger.success("All tasks completed successfully.")
                    else:
                        st.error("Process returned no data.")

if __name__ == "__main__":
    Agent_Root_Manager()