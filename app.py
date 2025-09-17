# 🚀 BRAND REPUTATION ANALYZER - VERSIONE ESPANSA
# Analisi completa della reputazione online tramite SERP e scraping multi-fonte

import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from difflib import get_close_matches
import warnings
import io
from collections import Counter
import numpy as np
from urllib.parse import urljoin, urlparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# 🎨 CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="🔍 Brand Reputation Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎯 CSS PERSONALIZZATO (stesso del codice originale)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .source-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .review-example {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    
    .positive-review {
        border-left: 4px solid #28a745;
    }
    
    .negative-review {
        border-left: 4px solid #dc3545;
    }
    
    .frequency-badge {
        background: #007bff;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .platform-badge {
        background: #6f42c1;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 HEADER PRINCIPALE
st.markdown('<h1 class="main-header">🔍 Brand Reputation Analyzer</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Cosa fa questa App Espansa?</h3>
    <p>• Ricerca automatica di tutte le fonti online dove il brand è recensito</p>
    <p>• Analisi SERP tramite API SERPER per trovare recensioni ovunque</p>
    <p>• Scraping multi-piattaforma: Trustpilot, Google Maps, Facebook, Amazon, Reddit, ecc.</p>
    <p>• Clustering intelligente delle recensioni per tematiche</p>
    <p>• Sentiment analysis completa con esempi verificabili</p>
    <p>• Report consolidato da tutte le fonti online</p>
    <p>• Strategie di Digital Marketing basate su analisi cross-platform</p>
</div>
""", unsafe_allow_html=True)

# 🔧 FUNZIONI BACKEND ESPANSE

class SerperClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query, num=20):
        """Ricerca tramite API SERPER"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num,
            'gl': 'it',  # Geolocalizzazione Italia
            'hl': 'it'   # Lingua italiana
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Errore API SERPER: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Errore connessione SERPER: {e}")
            return None

def genera_query_ricerca(brand_name):
    """Genera query di ricerca per trovare recensioni del brand"""
    query_templates = [
        f'"{brand_name}" recensioni',
        f'"{brand_name}" opinioni',
        f'"{brand_name}" review',
        f'"{brand_name}" feedback',
        f'"{brand_name}" esperienze',
        f'"{brand_name}" trustpilot',
        f'"{brand_name}" google reviews',
        f'"{brand_name}" facebook reviews',
        f'"{brand_name}" amazon recensioni',
        f'"{brand_name}" tripadvisor',
        f'"{brand_name}" reddit opinioni'
    ]
    return query_templates

def identifica_tipo_piattaforma(url):
    """Identifica il tipo di piattaforma da un URL"""
    url_lower = url.lower()
    
    platform_mapping = {
        'trustpilot.com': 'trustpilot',
        'google.com/maps': 'google_maps',
        'facebook.com': 'facebook',
        'amazon.it': 'amazon',
        'amazon.com': 'amazon',
        'reddit.com': 'reddit',
        'tripadvisor.it': 'tripadvisor',
        'tripadvisor.com': 'tripadvisor',
        'feedaty.com': 'feedaty',
        'opinioni.it': 'opinioni',
        'trovaprezzi.it': 'comparatore',
        'kelkoo.it': 'comparatore',
        'shopping.google': 'google_shopping'
    }
    
    for platform_url, platform_type in platform_mapping.items():
        if platform_url in url_lower:
            return platform_type
    
    return 'generic'

def estrai_recensioni_serp(serper_client, brand_name, progress_bar, status_text):
    """Estrae URL di recensioni dalle SERP"""
    fonti_trovate = []
    query_templates = genera_query_ricerca(brand_name)
    
    for i, query in enumerate(query_templates):
        status_text.text(f"🔍 Ricerca SERP: {query}")
        
        risultati = serper_client.search(query)
        if not risultati:
            continue
        
        # Elabora risultati organici
        for result in risultati.get('organic', []):
            url = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            if url and is_review_relevant(url, title, snippet, brand_name):
                platform_type = identifica_tipo_piattaforma(url)
                
                fonte = {
                    'url': url,
                    'title': title,
                    'snippet': snippet,
                    'platform': platform_type,
                    'query_source': query
                }
                
                # Evita duplicati
                if not any(f['url'] == url for f in fonti_trovate):
                    fonti_trovate.append(fonte)
        
        progress_bar.progress((i + 1) / len(query_templates))
        time.sleep(1)  # Rate limiting
    
    return fonti_trovate

def is_review_relevant(url, title, snippet, brand_name):
    """Verifica se un risultato è rilevante per le recensioni del brand"""
    relevance_keywords = [
        'recensioni', 'review', 'opinioni', 'feedback', 'esperienze', 
        'valutazioni', 'commenti', 'testimonianze'
    ]
    
    text_to_check = f"{title} {snippet}".lower()
    brand_lower = brand_name.lower()
    
    # Il brand deve essere presente
    if brand_lower not in text_to_check:
        return False
    
    # Deve contenere almeno una keyword di recensioni
    has_review_keyword = any(keyword in text_to_check for keyword in relevance_keywords)
    
    # Oppure deve essere una piattaforma di recensioni conosciuta
    is_review_platform = any(platform in url.lower() for platform in [
        'trustpilot', 'google.com/maps', 'facebook.com', 'amazon', 'reddit',
        'tripadvisor', 'feedaty', 'opinioni.it'
    ])
    
    return has_review_keyword or is_review_platform

def scrapa_contenuto_generico(url, platform_type='generic'):
    """Scraper generico per diversi tipi di piattaforme"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        recensioni = []
        
        if platform_type == 'trustpilot':
            recensioni = scrapa_trustpilot(soup, url)
        elif platform_type == 'facebook':
            recensioni = scrapa_facebook(soup, url)
        elif platform_type == 'google_maps':
            recensioni = scrapa_google_maps(soup, url)
        elif platform_type == 'amazon':
            recensioni = scrapa_amazon(soup, url)
        elif platform_type == 'reddit':
            recensioni = scrapa_reddit(soup, url)
        else:
            recensioni = scrapa_generico(soup, url)
        
        return recensioni
        
    except Exception as e:
        st.warning(f"Errore scraping {url}: {e}")
        return []

def scrapa_trustpilot(soup, url):
    """Scraper specifico per Trustpilot"""
    recensioni = []
    containers = soup.find_all('article', {'data-service-review-rating': True})
    
    for container in containers:
        try:
            testo = container.get_text(separator=" ", strip=True)
            rating_elem = container.get('data-service-review-rating')
            rating = int(rating_elem) if rating_elem else None
            
            if testo and len(testo) > 30:
                recensioni.append({
                    'testo': testo,
                    'rating': rating,
                    'source_url': url,
                    'platform': 'trustpilot'
                })
        except:
            continue
    
    return recensioni

def scrapa_facebook(soup, url):
    """Scraper per recensioni Facebook"""
    recensioni = []
    
    # Selettori comuni per recensioni Facebook
    selettori_recensioni = [
        '[data-testid*="review"]',
        '.review, .Review',
        '[role="article"]'
    ]
    
    for selettore in selettori_recensioni:
        containers = soup.select(selettore)
        
        for container in containers:
            testo = container.get_text(separator=" ", strip=True)
            
            if testo and len(testo) > 30 and 'recensione' in testo.lower():
                # Prova a estrarre rating se presente
                rating = None
                star_elements = container.find_all(['span', 'div'], class_=re.compile(r'star|rating'))
                for star_elem in star_elements:
                    star_text = star_elem.get_text()
                    rating_match = re.search(r'(\d)[^\d]*5', star_text)
                    if rating_match:
                        rating = int(rating_match.group(1))
                        break
                
                recensioni.append({
                    'testo': testo,
                    'rating': rating,
                    'source_url': url,
                    'platform': 'facebook'
                })
        
        if recensioni:  # Se trova recensioni con un selettore, non prova gli altri
            break
    
    return recensioni

def scrapa_google_maps(soup, url):
    """Scraper per Google Maps recensioni"""
    recensioni = []
    
    # Selettori per Google Maps
    selettori = [
        '[data-review-id]',
        '.review',
        '[jsaction*="review"]'
    ]
    
    for selettore in selettori:
        containers = soup.select(selettore)
        
        for container in containers:
            testo = container.get_text(separator=" ", strip=True)
            
            if testo and len(testo) > 30:
                # Estrai rating dalle stelle
                rating = None
                rating_elem = container.find(['span', 'div'], {'aria-label': re.compile(r'\d stelle')})
                if rating_elem:
                    rating_match = re.search(r'(\d) stelle', rating_elem.get('aria-label', ''))
                    if rating_match:
                        rating = int(rating_match.group(1))
                
                recensioni.append({
                    'testo': testo,
                    'rating': rating,
                    'source_url': url,
                    'platform': 'google_maps'
                })
    
    return recensioni

def scrapa_amazon(soup, url):
    """Scraper per recensioni Amazon"""
    recensioni = []
    
    containers = soup.find_all(['div'], {'data-hook': 'review'}) or soup.find_all(['div'], class_=re.compile(r'review'))
    
    for container in containers:
        testo = container.get_text(separator=" ", strip=True)
        
        if testo and len(testo) > 30:
            # Estrai rating
            rating = None
            rating_elem = container.find(['span'], class_=re.compile(r'star|rating'))
            if rating_elem:
                rating_match = re.search(r'(\d)[.,]?\d?.*su.*5', rating_elem.get_text())
                if rating_match:
                    rating = int(rating_match.group(1))
            
            recensioni.append({
                'testo': testo,
                'rating': rating,
                'source_url': url,
                'platform': 'amazon'
            })
    
    return recensioni

def scrapa_reddit(soup, url):
    """Scraper per post Reddit"""
    recensioni = []
    
    # Cerca post e commenti
    containers = soup.find_all(['div'], class_=re.compile(r'Post|Comment'))
    containers.extend(soup.find_all(['p']))
    
    for container in containers:
        testo = container.get_text(separator=" ", strip=True)
        
        if testo and len(testo) > 50:
            recensioni.append({
                'testo': testo,
                'rating': None,  # Reddit non ha rating standard
                'source_url': url,
                'platform': 'reddit'
            })
    
    return recensioni

def scrapa_generico(soup, url):
    """Scraper generico per siti non specifici"""
    recensioni = []
    
    # Cerca contenuto che potrebbe essere recensioni
    contenuti_potenziali = []
    
    # Paragraphs
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        if len(text) > 50:
            contenuti_potenziali.append(text)
    
    # Divs con classi che suggeriscono recensioni
    for div in soup.find_all('div', class_=re.compile(r'review|comment|feedback|opinion', re.I)):
        text = div.get_text(separator=" ", strip=True)
        if len(text) > 50:
            contenuti_potenziali.append(text)
    
    # Filtra contenuti che sembrano recensioni
    for testo in contenuti_potenziali:
        if sembra_recensione(testo):
            recensioni.append({
                'testo': testo,
                'rating': estrai_rating_da_testo_generico(testo),
                'source_url': url,
                'platform': 'generic'
            })
    
    return recensioni

def sembra_recensione(testo):
    """Verifica se un testo sembra una recensione"""
    indicatori_recensione = [
        'esperienza', 'consiglio', 'sconsiglio', 'ottimo', 'pessimo', 
        'soddisfatto', 'deluso', 'acquistato', 'servizio', 'prodotto',
        'stelle', 'voto', 'rating', 'giudizio'
    ]
    
    testo_lower = testo.lower()
    count = sum(1 for indicatore in indicatori_recensione if indicatore in testo_lower)
    
    return count >= 2 and len(testo) > 30

def estrai_rating_da_testo_generico(testo):
    """Estrae rating da testo generico"""
    # Cerca pattern di rating
    patterns = [
        r'(\d)\s*su\s*5',
        r'(\d)\s*stelle',
        r'voto\s*(\d)',
        r'rating\s*(\d)',
        r'(\d)/5'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, testo, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None

def pulisci_testo_espanso(testo):
    """Versione espansa della pulizia del testo"""
    # Utilizza la funzione originale come base
    stopwords_italiane = get_stopwords()
    testo = testo.lower()
    
    # Rimozioni specifiche per diverse piattaforme
    testo = re.sub(r'pubblicat[oae]\s+il\s+\d{1,2}\s+\w+\s+\d{4}', '', testo)
    testo = re.sub(r'data\s+dell[\'']?esperienza:.*', '', testo, flags=re.IGNORECASE)
    testo = re.sub(r'verificat[aoe]', '', testo)
    testo = re.sub(r'\d+\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}', '', testo)
    
    # Rimozioni Facebook/social
    testo = re.sub(r'mi\s+piace|like|share|condividi', '', testo)
    testo = re.sub(r'\d+\s+(ore|giorni|settimane|mesi)\s+fa', '', testo)
    
    # Rimozioni Amazon
    testo = re.sub(r'acquisto\s+verificato|vine\s+customer', '', testo)
    testo = re.sub(r'helpful|utile\s+\d+', '', testo)
    
    # Pulizia generale
    testo = re.sub(r'[^\w\s]', ' ', testo)
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    
    parole = testo.split()
    parole_filtrate = [parola for parola in parole if parola not in stopwords_italiane and len(parola) > 2]
    return " ".join(parole_filtrate)

@st.cache_data
def get_stopwords():
    """Stopwords italiane espanse"""
    return set([
        "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su", "per", 
        "tra", "fra", "un", "una", "uno", "e", "ma", "anche", "come", "che", "non", 
        "più", "meno", "molto", "poco", "tutto", "tutti", "tutte", "questo", "questa", 
        "questi", "queste", "quello", "quella", "quelli", "quelle", "sono", "è", "ho", 
        "hai", "ha", "hanno", "essere", "avere", "fare", "dire", "andare", "del", "della",
        "dei", "delle", "dal", "dalla", "dai", "dalle", "nel", "nella", "nei", "nelle",
        "sul", "sulla", "sui", "sulle", "al", "alla", "ai", "alle", "ho", "ottimo",
        "buono", "buona", "bene", "male", "servizio", "prodotto", "azienda", "sempre",
        "verificata", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
        "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
        "facebook", "google", "maps", "amazon", "reddit", "trustpilot"
    ])

def elabora_fonti_parallele(fonti_trovate, max_workers=5, progress_bar=None, status_text=None):
    """Elabora le fonti in parallelo per velocizzare il processo"""
    tutte_recensioni = []
    statistiche_fonti = {}
    
    def scrapa_fonte(fonte):
        url = fonte['url']
        platform = fonte['platform']
        
        try:
            recensioni = scrapa_contenuto_generico(url, platform)
            return {
                'fonte': fonte,
                'recensioni': recensioni,
                'successo': True
            }
        except Exception as e:
            return {
                'fonte': fonte,
                'recensioni': [],
                'successo': False,
                'errore': str(e)
            }
    
    # Processa in parallelo con ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fonte = {executor.submit(scrapa_fonte, fonte): fonte for fonte in fonti_trovate}
        
        for i, future in enumerate(as_completed(future_to_fonte)):
            if status_text:
                status_text.text(f"🕷️ Scraping fonte {i+1}/{len(fonti_trovate)}...")
            
            if progress_bar:
                progress_bar.progress((i + 1) / len(fonti_trovate))
            
            risultato = future.result()
            fonte = risultato['fonte']
            recensioni = risultato['recensioni']
            
            if risultato['successo'] and recensioni:
                # Arricchisci le recensioni con metadata
                for rec in recensioni:
                    rec['testo_pulito'] = pulisci_testo_espanso(rec['testo'])
                    rec['fonte_title'] = fonte['title']
                    rec['fonte_snippet'] = fonte['snippet']
                
                tutte_recensioni.extend(recensioni)
                statistiche_fonti[fonte['platform']] = statistiche_fonti.get(fonte['platform'], 0) + len(recensioni)
            
            time.sleep(0.5)  # Rate limiting
    
    return tutte_recensioni, statistiche_fonti

def analizza_multi_fonte_con_ai(recensioni_multi_fonte, client, progress_bar, status_text):
    """Analisi AI delle recensioni multi-fonte"""
    # Organizza recensioni per piattaforma
    recensioni_per_piattaforma = {}
    for rec in recensioni_multi_fonte:
        platform = rec.get('platform', 'unknown')
        if platform not in recensioni_per_piattaforma:
            recensioni_per_piattaforma[platform] = []
        recensioni_per_piattaforma[platform].append(rec)
    
    # Prepara testi per analisi
    testi_puliti = [rec['testo_pulito'] for rec in recensioni_multi_fonte if rec['testo_pulito']]
    
    if not testi_puliti:
        return {}
    
    # Dividi in blocchi
    testo_completo = " ".join(testi_puliti)
    parole = testo_completo.split()
    blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
    
    risultati = {
        "punti_forza": [],
        "punti_debolezza": [],
        "leve_marketing": [],
        "parole_chiave": [],
        "suggerimenti_seo": [],
        "suggerimenti_adv": [],
        "suggerimenti_email": [],
        "suggerimenti_cro": [],
        "suggerimenti_sinergie": [],
        "analisi_per_piattaforma": {},
        "sentiment_distribution": {"positivo": 0, "neutro": 0, "negativo": 0}
    }
    
    # Analisi generale
    for i, blocco in enumerate(blocchi):
        status_text.text(f"🤖 Analizzando blocco {i+1}/{len(blocchi)} con AI...")
        
        prompt = f"""
        Analizza le seguenti recensioni multi-piattaforma di un brand e fornisci insights strategici:

        RECENSIONI DA DIVERSE FONTI:
        {blocco}

        Rispondi SOLO in formato JSON valido con queste chiavi:
        {{
            "punti_forza": ["punto specifico 1", "punto specifico 2", ...],
            "punti_debolezza": ["problema specifico 1", "problema specifico 2", ...],
            "leve_marketing": ["leva concreta 1", "leva concreta 2", ...],
            "parole_chiave": ["termine rilevante 1", "termine rilevante 2", ...],
            "suggerimenti_seo": ["suggerimento SEO specifico 1", "suggerimento SEO specifico 2", ...],
            "suggerimenti_adv": ["strategia pubblicitaria 1", "strategia pubblicitaria 2", ...],
            "suggerimenti_email": ["strategia email 1", "strategia email 2", ...],
            "suggerimenti_cro": ["ottimizzazione conversioni 1", "ottimizzazione conversioni 2", ...],
            "suggerimenti_sinergie": ["sinergia multi-canale 1", "sinergia multi-canale 2", ...],
            "sentiment_counts": {{"positivo": N, "neutro": N, "negativo": N}}
        }}

        IMPORTANTE:
        - Considera che le recensioni provengono da piattaforme diverse (Trustpilot, Google, Facebook, ecc.)
        - Identifica pattern cross-platform e differenze tra piattaforme
        - Le sinergie devono considerare la presenza multi-piattaforma del brand
        - Ignora termini temporali e riferimenti alle piattaforme stesse
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            content_cleaned = re.sub(r"```json\n?|```", "", content).strip()
            dati = json.loads(content_cleaned)
            
            # Unisci i risultati
            for chiave in risultati:
                if chiave in dati:
                    if chiave == "sentiment_counts":
                        for sent_type in ['positivo', 'neutro', 'negativo']:
                            if sent_type in dati['sentiment_counts']:
                                risultati['sentiment_distribution'][sent_type] += dati['sentiment_counts'][sent_type]
                    elif chiave != "analisi_per_piattaforma":
                        nuovi_elementi = [elem for elem in dati[chiave] if elem not in risultati[chiave]]
                        risultati[chiave].extend(nuovi_elementi)
        
        except Exception as e:
            st.warning(f"Errore analisi AI blocco {i+1}: {e}")
        
        progress_bar.progress((i + 1) / len(blocchi))
    
    # Analisi specifica per piattaforma
    for platform, recensioni_platform in recensioni_per_piattaforma.items():
        if len(recensioni_platform) >= 5:  # Solo se ci sono abbastanza recensioni
            testi_platform = [rec['testo_pulito'] for rec in recensioni_platform]
            sample_text = " ".join(testi_platform[:50])  # Campione per analisi
            
            risultati["analisi_per_piattaforma"][platform] = {
                'n_recensioni': len(recensioni_platform),
                'rating_medio': np.mean([rec['rating'] for rec in recensioni_platform if rec['rating']]),
                'temi_specifici': estrai_temi_specifici(sample_text)
            }
    
    return risultati

def estrai_temi_specifici(testo, n_temi=5):
    """Estrae temi specifici da un testo usando TF-IDF"""
    if not testo or len(testo.split()) < 10:
        return []
    
    try:
        vectorizer = TfidfVectorizer(max_features=20, min_df=1, token_pattern=r'\b[a-zA-Z]{3,}\b')
        tfidf_matrix = vectorizer.fit_transform([testo])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Ottieni top temi
        top_indices = scores.argsort()[-n_temi:][::-1]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except:
        return []

def crea_report_multi_fonte(recensioni_data, risultati, statistiche_fonti):
    """Crea report Excel multi-fonte"""
    output = io.BytesIO()
    
    # DataFrame recensioni con fonte
    df_recensioni = pd.DataFrame([{
        'Testo': rec['testo'],
        'Rating': rec.get('rating', 'N/A'),
        'Piattaforma': rec['platform'],
        'URL Fonte': rec['source_url'],
        'Titolo Fonte': rec.get('fonte_title', ''),
        'Snippet': rec.get('fonte_snippet', '')
    } for rec in recensioni_data])
    
    # DataFrame statistiche per piattaforma
    df_statistiche = pd.DataFrame([{
        'Piattaforma': platform,
        'N. Recensioni': count,
        'Percentuale': f"{(count/len(recensioni_data)*100):.1f}%"
    } for platform, count in statistiche_fonti.items()])
    
    # DataFrame analisi per piattaforma
    platform_analysis = []
    for platform, analisi in risultati.get('analisi_per_piattaforma', {}).items():
        platform_analysis.append({
            'Piattaforma': platform,
            'N. Recensioni': analisi['n_recensioni'],
            'Rating Medio': f"{analisi['rating_medio']:.1f}" if not np.isnan(analisi['rating_medio']) else 'N/A',
            'Temi Specifici': ', '.join(analisi['temi_specifici'])
        })
    
    df_platform_analysis = pd.DataFrame(platform_analysis)
    
    # Altri DataFrame (come nel codice originale)
    df_insights = pd.DataFrame({
        'Categoria': ['Punti Forza', 'Punti Debolezza', 'Leve Marketing', 'Parole Chiave'],
        'Insights': [
            ' | '.join(risultati.get('punti_forza', [])[:10]),
            ' | '.join(risultati.get('punti_debolezza', [])[:10]),
            ' | '.join(risultati.get('leve_marketing', [])[:10]),
            ' | '.join(risultati.get('parole_chiave', [])[:20])
        ]
    })
    
    df_strategie = pd.DataFrame({
        'Canale': ['SEO', 'ADV', 'Email Marketing', 'CRO', 'Sinergie Multi-Platform'],
        'Suggerimenti': [
            ' | '.join(risultati.get('suggerimenti_seo', [])),
            ' | '.join(risultati.get('suggerimenti_adv', [])),
            ' | '.join(risultati.get('suggerimenti_email', [])),
            ' | '.join(risultati.get('suggerimenti_cro', [])),
            ' | '.join(risultati.get('suggerimenti_sinergie', []))
        ]
    })
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_recensioni.to_excel(writer, sheet_name='Tutte le Recensioni', index=False)
        df_statistiche.to_excel(writer, sheet_name='Statistiche per Piattaforma', index=False)
        df_platform_analysis.to_excel(writer, sheet_name='Analisi per Piattaforma', index=False)
        df_insights.to_excel(writer, sheet_name='Insights Generali', index=False)
        df_strategie.to_excel(writer, sheet_name='Strategie Digital', index=False)
    
    return output.getvalue()

# 🎮 INTERFACCIA PRINCIPALE ESPANSA
def main():
    # SIDEBAR ESPANSA
    with st.sidebar:
        st.markdown("## 🔧 Configurazione Espansa")
        
        # API Keys
        st.markdown("### 🔑 API Keys")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="API Key di OpenAI per l'analisi AI"
        )
        
        serper_key = st.text_input(
            "SERPER API Key",
            type="password",
            help="API Key di SERPER per la ricerca nelle SERP"
        )
        
        st.markdown("### 🏢 Brand Information")
        
        # Nome Brand
        brand_name = st.text_input(
            "Nome del Brand",
            placeholder="es: NomeBrand S.r.l.",
            help="Nome completo del brand da analizzare"
        )
        
        # URL Sito (opzionale)
        sito_url = st.text_input(
            "URL Sito Web (opzionale)",
            placeholder="https://www.nomebrand.com",
            help="URL del sito web del brand"
        )
        
        st.markdown("### 🎛️ Parametri Avanzati")
        
        # Numero risultati SERP
        max_risultati_serp = st.slider(
            "Risultati SERP per query",
            min_value=10,
            max_value=50,
            value=20,
            help="Numero di risultati da analizzare per ogni ricerca"
        )
        
        # Workers paralleli
        max_workers = st.slider(
            "Processi paralleli",
            min_value=1,
            max_value=10,
            value=5,
            help="Numero di scraper paralleli (più alto = più veloce ma più carico)"
        )
        
        # Filtro piattaforme
        st.markdown("### 📱 Piattaforme da Includere")
        piattaforme_target = st.multiselect(
            "Seleziona piattaforme",
            options=['trustpilot', 'google_maps', 'facebook', 'amazon', 'reddit', 'tripadvisor', 'feedaty', 'opinioni', 'generic'],
            default=['trustpilot', 'google_maps', 'facebook', 'amazon', 'reddit'],
            help="Piattaforme su cui cercare recensioni"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Suggerimenti Espansi")
        st.info("""
        **Nuovo Workflow:**
        1. Inserisci nome brand e API keys
        2. L'app cerca automaticamente recensioni ovunque online
        3. Analizza sentiment cross-platform
        4. Genera strategie basate su tutti i dati
        
        **API Keys Richieste:**
        - OpenAI: Per analisi AI avanzata
        - SERPER: Per ricerca automatica nelle SERP
        
        **Tempo stimato:** 10-20 minuti per analisi completa
        """)

    # AREA PRINCIPALE ESPANSA
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Analisi Reputazione Completa")
        
        if st.button("🔍 Avvia Analisi Multi-Piattaforma", type="primary"):
            # Validazione input
            errori = []
            if not openai_key:
                errori.append("OpenAI API Key mancante")
            if not serper_key:
                errori.append("SERPER API Key mancante")
            if not brand_name:
                errori.append("Nome brand mancante")
            
            if errori:
                for errore in errori:
                    st.error(f"❌ {errore}")
                return
            
            # Inizializza client
            try:
                openai_client = OpenAI(api_key=openai_key)
                serper_client = SerperClient(serper_key)
            except Exception as e:
                st.error(f"❌ Errore inizializzazione client: {e}")
                return
            
            # Container risultati
            results_container = st.container()
            
            with results_container:
                # FASE 1: Ricerca SERP
                st.markdown("### 🔍 Fase 1: Ricerca Automatica nelle SERP")
                progress_bar_1 = st.progress(0)
                status_text_1 = st.empty()
                
                with st.spinner("Ricerca automatica di tutte le fonti online..."):
                    fonti_trovate = estrai_recensioni_serp(serper_client, brand_name, progress_bar_1, status_text_1)
                
                if not fonti_trovate:
                    st.error("❌ Nessuna fonte di recensioni trovata per il brand")
                    return
                
                # Filtra per piattaforme selezionate
                fonti_filtrate = [f for f in fonti_trovate if f['platform'] in piattaforme_target]
                
                st.success(f"✅ Trovate {len(fonti_filtrate)} fonti rilevanti di recensioni!")
                
                # Mostra fonti trovate
                with st.expander("👀 Fonti Trovate"):
                    for fonte in fonti_filtrate[:10]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{fonte['title']}</strong>
                            <span class="platform-badge">{fonte['platform']}</span>
                            <br>
                            <small>{fonte['snippet']}</small>
                            <br>
                            <a href="{fonte['url']}" target="_blank">🔗 {fonte['url']}</a>
                        </div>
                        """, unsafe_allow_html=True)
                
                # FASE 2: Scraping Multi-Fonte
                st.markdown("### 🕷️ Fase 2: Scraping Multi-Piattaforma")
                progress_bar_2 = st.progress(0)
                status_text_2 = st.empty()
                
                with st.spinner("Estrazione recensioni da tutte le fonti..."):
                    recensioni_multi_fonte, statistiche_fonti = elabora_fonti_parallele(
                        fonti_filtrate, max_workers, progress_bar_2, status_text_2
                    )
                
                if not recensioni_multi_fonte:
                    st.error("❌ Nessuna recensione estratta dalle fonti")
                    return
                
                st.success(f"✅ Estratte {len(recensioni_multi_fonte)} recensioni da {len(statistiche_fonti)} piattaforme!")
                
                # Statistiche piattaforme
                st.markdown("### 📊 Distribuzione per Piattaforma")
                col_stats = st.columns(len(statistiche_fonti))
                for i, (platform, count) in enumerate(statistiche_fonti.items()):
                    with col_stats[i % len(col_stats)]:
                        st.metric(platform.title(), count)
                
                # FASE 3: Analisi AI Multi-Fonte
                st.markdown("### 🤖 Fase 3: Analisi AI Cross-Platform")
                progress_bar_3 = st.progress(0)
                status_text_3 = st.empty()
                
                with st.spinner("Analisi AI avanzata multi-piattaforma..."):
                    risultati = analizza_multi_fonte_con_ai(
                        recensioni_multi_fonte, openai_client, progress_bar_3, status_text_3
                    )
                
                st.markdown('<div class="success-box"><h3>🎉 Analisi Multi-Piattaforma Completata!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI ESPANSI
                st.markdown("## 📊 Risultati Analisi Cross-Platform")
                
                # Metriche principali espanse
                col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
                
                with col_m1:
                    st.metric("📝 Tot. Recensioni", len(recensioni_multi_fonte))
                
                with col_m2:
                    ratings_validi = [r['rating'] for r in recensioni_multi_fonte if r.get('rating')]
                    rating_medio = np.mean(ratings_validi) if ratings_validi else 0
                    st.metric("⭐ Rating Medio", f"{rating_medio:.1f}")
                
                with col_m3:
                    st.metric("📱 Piattaforme", len(statistiche_fonti))
                
                with col_m4:
                    st.metric("💪 Punti Forza", len(risultati.get('punti_forza', [])))
                
                with col_m5:
                    st.metric("⚠️ Criticità", len(risultati.get('punti_debolezza', [])))
                
                with col_m6:
                    st.metric("🎯 Fonti Online", len(fonti_filtrate))
                
                # Distribuzione sentiment
                sentiment_dist = risultati.get('sentiment_distribution', {})
                if any(sentiment_dist.values()):
                    st.markdown("### 😊 Distribuzione Sentiment Multi-Platform")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    total = sum(sentiment_dist.values())
                    
                    with col_s1:
                        perc = (sentiment_dist.get('positivo', 0) / total * 100) if total > 0 else 0
                        st.metric("😊 Positivo", f"{perc:.1f}%")
                    
                    with col_s2:
                        perc = (sentiment_dist.get('neutro', 0) / total * 100) if total > 0 else 0
                        st.metric("😐 Neutro", f"{perc:.1f}%")
                    
                    with col_s3:
                        perc = (sentiment_dist.get('negativo', 0) / total * 100) if total > 0 else 0
                        st.metric("😞 Negativo", f"{perc:.1f}%")
                
                # Tabs espanse
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "💪 Punti Forza", 
                    "⚠️ Criticità", 
                    "📱 Per Piattaforma", 
                    "🎯 Leve Marketing", 
                    "📈 Strategie Digital", 
                    "🔍 Keywords",
                    "🔗 Fonti Verificabili"
                ])
                
                with tab1:
                    st.markdown("### 💪 Punti di Forza Cross-Platform")
                    for i, punto in enumerate(risultati.get('punti_forza', [])[:15], 1):
                        st.markdown(f"**{i}.** {punto}")
                
                with tab2:
                    st.markdown("### ⚠️ Punti di Debolezza Cross-Platform")
                    for i, punto in enumerate(risultati.get('punti_debolezza', [])[:15], 1):
                        st.markdown(f"**{i}.** {punto}")
                
                with tab3:
                    st.markdown("### 📱 Analisi Specifica per Piattaforma")
                    
                    for platform, analisi in risultati.get('analisi_per_piattaforma', {}).items():
                        with st.expander(f"{platform.title()} - {analisi['n_recensioni']} recensioni"):
                            col_p1, col_p2 = st.columns(2)
                            
                            with col_p1:
                                st.metric("Recensioni", analisi['n_recensioni'])
                                rating = analisi.get('rating_medio', 0)
                                if not np.isnan(rating) and rating > 0:
                                    st.metric("Rating Medio", f"{rating:.1f} ⭐")
                            
                            with col_p2:
                                st.markdown("**Temi Specifici:**")
                                for tema in analisi.get('temi_specifici', []):
                                    st.markdown(f"• {tema}")
                            
                            # Mostra alcune recensioni esempio per questa piattaforma
                            recensioni_piattaforma = [r for r in recensioni_multi_fonte if r['platform'] == platform][:3]
                            st.markdown("**Recensioni Esempio:**")
                            for rec in recensioni_piattaforma:
                                rating_str = f"⭐{rec['rating']}" if rec.get('rating') else "N/A"
                                testo_breve = rec['testo'][:200] + "..." if len(rec['testo']) > 200 else rec['testo']
                                st.markdown(f"> **{rating_str}** {testo_breve}")
                                st.markdown(f"[🔗 Fonte]({rec['source_url']})")
                
                with tab4:
                    st.markdown("### 🎯 Leve Marketing Cross-Platform")
                    for i, leva in enumerate(risultati.get('leve_marketing', [])[:12], 1):
                        st.markdown(f"**{i}.** {leva}")
                
                with tab5:
                    st.markdown("### 📈 Strategie Digital Multi-Platform")
                    
                    col_seo, col_adv = st.columns(2)
                    
                    with col_seo:
                        st.markdown("#### 🌐 SEO Multi-Platform")
                        for sug in risultati.get('suggerimenti_seo', [])[:6]:
                            st.markdown(f"• {sug}")
                        
                        st.markdown("#### 📧 Email Marketing")
                        for sug in risultati.get('suggerimenti_email', [])[:6]:
                            st.markdown(f"• {sug}")
                    
                    with col_adv:
                        st.markdown("#### 📢 ADV Cross-Platform")
                        for sug in risultati.get('suggerimenti_adv', [])[:6]:
                            st.markdown(f"• {sug}")
                        
                        st.markdown("#### 🔄 CRO")
                        for sug in risultati.get('suggerimenti_cro', [])[:6]:
                            st.markdown(f"• {sug}")
                    
                    st.markdown("#### 🤝 Sinergie Multi-Platform")
                    for sug in risultati.get('suggerimenti_sinergie', [])[:8]:
                        st.markdown(f"• {sug}")
                
                with tab6:
                    st.markdown("### 🔍 Keywords Cross-Platform")
                    keywords_cols = st.columns(4)
                    for i, keyword in enumerate(risultati.get('parole_chiave', [])[:20]):
                        with keywords_cols[i % 4]:
                            st.markdown(f"🔸 **{keyword}**")
                
                with tab7:
                    st.markdown("### 🔗 Fonti Verificabili per Categoria")
                    
                    # Raggruppa fonti per piattaforma
                    fonti_per_piattaforma = {}
                    for fonte in fonti_filtrate:
                        platform = fonte['platform']
                        if platform not in fonti_per_piattaforma:
                            fonti_per_piattaforma[platform] = []
                        fonti_per_piattaforma[platform].append(fonte)
                    
                    for platform, fonti_platform in fonti_per_piattaforma.items():
                        with st.expander(f"{platform.title()} - {len(fonti_platform)} fonti"):
                            for fonte in fonti_platform:
                                st.markdown(f"""
                                **{fonte['title']}**  
                                {fonte['snippet']}  
                                [🔗 Visita fonte]({fonte['url']})
                                """)
                
                # DOWNLOAD ESPANSO
                st.markdown("## 📥 Download Report Multi-Platform")
                
                excel_data = crea_report_multi_fonte(recensioni_multi_fonte, risultati, statistiche_fonti)
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.download_button(
                        label="📊 Scarica Report Excel Multi-Platform",
                        data=excel_data,
                        file_name=f"Analisi_Brand_Reputation_{brand_name.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                
                with col_d2:
                    # Report JSON espanso
                    json_report = {
                        'metadata': {
                            'brand_name': brand_name,
                            'sito_url': sito_url,
                            'n_recensioni_totali': len(recensioni_multi_fonte),
                            'n_piattaforme': len(statistiche_fonti),
                            'n_fonti': len(fonti_filtrate),
                            'rating_medio_globale': float(rating_medio) if rating_medio else None
                        },
                        'statistiche_piattaforme': statistiche_fonti,
                        'analisi_per_piattaforma': risultati.get('analisi_per_piattaforma', {}),
                        'insights_globali': {
                            'punti_forza': risultati.get('punti_forza', [])[:20],
                            'punti_debolezza': risultati.get('punti_debolezza', [])[:20],
                            'leve_marketing': risultati.get('leve_marketing', [])[:15],
                            'parole_chiave': risultati.get('parole_chiave', [])[:30]
                        },
                        'strategie_digital': {
                            'seo': risultati.get('suggerimenti_seo', []),
                            'adv': risultati.get('suggerimenti_adv', []),
                            'email': risultati.get('suggerimenti_email', []),
                            'cro': risultati.get('suggerimenti_cro', []),
                            'sinergie': risultati.get('suggerimenti_sinergie', [])
                        },
                        'sentiment_distribution': risultati.get('sentiment_distribution', {}),
                        'fonti_verificabili': [{'url': f['url'], 'title': f['title'], 'platform': f['platform']} 
                                              for f in fonti_filtrate[:50]]
                    }
                    
                    st.download_button(
                        label="💾 Scarica Report JSON Espanso",
                        data=json.dumps(json_report, indent=2, ensure_ascii=False),
                        file_name=f"Analisi_Brand_Data_{brand_name.replace(' ', '_')}.json",
                        mime="application/json"
                    )
    
    with col2:
        st.markdown("## 📋 Guida Espansa")
        
        st.markdown("""
        ### 🆕 Funzionalità Multi-Platform:
        - **🔍 Ricerca SERP Automatica**: Trova tutte le fonti di recensioni online
        - **🕷️ Scraping Multi-Fonte**: Estrae da Trustpilot, Google, Facebook, Amazon, Reddit, ecc.
        - **📊 Analisi Cross-Platform**: Confronta sentiment tra diverse piattaforme
        - **🔗 Fonti Verificabili**: Link diretti a tutte le recensioni analizzate
        - **📱 Insights per Piattaforma**: Analisi specifica per ogni fonte
        - **🤝 Sinergie Multi-Channel**: Strategie che sfruttano tutte le piattaforme
        
        ### 🎯 Workflow Completo:
        1. **Setup**: Inserisci brand name e API keys
        2. **Discovery**: L'app trova automaticamente tutte le fonti online
        3. **Extraction**: Scraping parallelo da tutte le piattaforme
        4. **Analysis**: AI analizza sentiment e temi cross-platform
        5. **Insights**: Report completo con strategie specifiche
        6. **Export**: Download dati per ulteriori elaborazioni
        
        ### 📈 Output Multi-Dimensionale:
        • **Coverage totale** della presenza online
        • **Comparison** tra piattaforme diverse
        • **Trend identification** cross-platform
        • **Strategic recommendations** basate su tutti i dati
        • **Actionable insights** per ogni canale
        • **Verifiable sources** per ogni insight
        
        ### 🔑 API Keys Richieste:
        **SERPER**: [serper.dev](https://serper.dev)
        - Piano gratuito: 2500 query/mese
        - Costo: $5 per 10k query
        
        **OpenAI**: [platform.openai.com](https://platform.openai.com)
        - Costo variabile per token utilizzati
        """)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;'>
            <h4>⚡ Performance</h4>
            <p><strong>Tempo stimato:</strong><br>
            • Setup: 2-3 minuti<br>
            • Discovery SERP: 3-5 minuti<br>
            • Multi-platform scraping: 10-15 minuti<br>
            • AI Analysis: 5-8 minuti<br>
            <strong>Totale: 20-30 minuti</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Footer espanso
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Brand Reputation Analyzer v3.0 - Multi-Platform Edition</strong></p>
        <p>Analisi completa della reputazione online con ricerca automatica SERP e scraping multi-fonte</p>
        <p>Sviluppato con ❤️ utilizzando OpenAI GPT-4 & SERPER API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
