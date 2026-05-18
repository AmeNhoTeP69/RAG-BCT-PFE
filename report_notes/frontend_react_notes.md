# Frontend React — Notes Techniques pour la Soutenance

**Date:** Mai 2026
**Projet:** BCT Intel-Graph — Interface Hybrid Graph RAG
**Technologie:** React 19 + Vite 8 + FastAPI (Option A — Serveur Unique)

---

## 1. Architecture Technique du Frontend

### Stack Technologique
| Couche | Technologie | Justification |
|---|---|---|
| UI Framework | **React 19** | Composants réutilisables, gestion d'état réactive |
| Build Tool | **Vite 8** | Build ultra-rapide (800ms), HMR en développement |
| Rendu Markdown | **react-markdown** | Rendu natif des réponses formatées du LLM |
| Icônes | **lucide-react** | Bibliothèque d'icônes SVG légère et cohérente |
| CSS | **Vanilla CSS** | Design tokens, glassmorphism, animations personnalisées |

### Intégration Backend (Option A — Serveur Unique)
```
┌────────────────────────────────────────────────────────┐
│  Navigateur (localhost:8000)                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  React App (SPA — static/index.html)            │  │
│  │  POST /api/query ──────────────────────────────►│  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI (api.py) — Port 8000                          │
│  ├─ GET / ──► static/index.html (React)                │
│  ├─ GET /assets/* ──► static/assets/ (JS/CSS)          │
│  ├─ POST /api/query ──► RAG Engine ou Graph Engine     │
│  └─ GET /{any} ──► static/index.html (SPA fallback)   │
└────────────────────────────────────────────────────────┘
```

**Commande de démarrage (une seule) :**
```bash
python api.py
# Accès : http://localhost:8000
```

---

## 2. Design System — "BCT Midnight"

### Palette de Couleurs
```css
--bg-deep:    #080b14   /* Fond principal — Bleu nuit profond  */
--accent:     #D4AF37   /* Or BCT — Autorité de la Banque      */
--blue-graph: #4f8ef7   /* Bleu graphe — Visualisation réseau  */
--text:       #f0f0f5   /* Texte principal — Contraste élevé   */
```

### Effets Visuels
- **Glassmorphism** : `backdrop-filter: blur(18px)` sur les panneaux
- **Orbes de fond** : Dégradés radiaux animés pour la profondeur
- **Animations** : `fade-slide-up`, `thinking-bounce`, `pulse-dot`
- **RTL Automatique** : Détection de l'arabe par regex (`/[\u0600-\u06FF]/`)

### Typographie
- **Police** : Inter (Google Fonts) — Lisibilité professionnelle
- **Hiérarchie** : 10px (labels) → 22px (titres)

---

## 3. Composants React — Architecture

```
App.jsx                          ← État global (mode, messages, sélection)
├── Sidebar.jsx                  ← Sidebar gauche (280px)
│   ├── Brand (Logo + Titre)
│   ├── Mode Toggle (RAG / Graph RAG)
│   ├── Stats du Corpus (450+ docs, 2.3k relations)
│   └── Historique des requêtes
├── ChatWindow.jsx               ← Zone centrale (flex: 1)
│   ├── TopNav (Titre + Status)
│   ├── MessagesList
│   │   ├── WelcomeCard (état vide)
│   │   ├── Message.jsx (bulles user/IA)
│   │   └── ThinkingBubble.jsx (loading)
│   └── InputArea (textarea + bouton)
└── SourcePanel.jsx              ← Panneau droit (340px)
    ├── Sources (cartes documents)
    ├── Graph Nodes (badges réseau)
    └── Evidence (extraits textuels)
```

### Gestion d'État
```javascript
// App.jsx — État centralisé
const [mode, setMode] = useState('rag')     // 'rag' | 'graph'
const [messages, setMessages] = useState([]) // historique
const [isLoading, setIsLoading] = useState(false)
const [selectedMsg, setSelectedMsg] = useState(null) // panneau droit
```

---

## 4. Fonctionnalités Clés pour la Démonstration

### 4.1 Basculement Standard RAG ↔ Graph RAG
- Bouton dans la barre latérale gauche
- Indicateur visuel actif (or = RAG, bleu = Graph)
- Description contextuelle du mode sélectionné

### 4.2 Panneau des Preuves (Right Panel)
- **Sources** : Liste des circulaires utilisées
- **Entités Graphe** : Nœuds extraits du Knowledge Graph (mode Graph uniquement)
- **Extraits** : Texte brut des chunks documentaires avec numéro d'article

### 4.3 Support Multilingue Automatique
- Rendu RTL automatique (arabe)
- Placeholder trilingue : Français, العربية, English
- Exemples de requêtes dans les 3 langues sur l'écran d'accueil

### 4.4 Expérience Utilisateur Premium
- Animation "Analyse en cours..." pendant le traitement
- Chips d'exemples cliquables sur l'écran vide
- Historique des requêtes dans la sidebar
- Auto-scroll vers la dernière réponse
- Textarea auto-redimensionnable

---

## 5. Build & Déploiement

### Workflow de Build
```bash
cd frontend
npm run build
# → Génère : ../static/index.html + ../static/assets/
# → Taille : ~322 KB JS (100 KB gzip)
```

### Développement (Hot-Reload)
```bash
# Terminal 1
python api.py          # Backend port 8000

# Terminal 2
cd frontend
npm run dev            # Frontend port 5173 (avec proxy /api)
```

---

## 6. Points Clés à Présenter lors de la Soutenance

1. **"Serveur unique"** : Un seul `python api.py` suffit — pas de configuration complexe
2. **"React en production"** : Le build Vite optimise et compresse tout en 800ms
3. **"Panneaux dynamiques"** : Le panneau de preuves change selon la réponse cliquée — visualisation de la "preuve documentaire"
4. **"RTL natif"** : Détection automatique de l'arabe — prouve le support multilingue réel
5. **"Mode Graph visible"** : En mode Graph RAG, les nœuds du graphe de connaissances apparaissent sous la réponse — preuve visuelle du traversal

---

## 7. Statistiques Techniques (Pour le Rapport)

| Métrique | Valeur |
|---|---|
| Composants React | 6 (App, Sidebar, ChatWindow, Message, SourcePanel, ThinkingBubble) |
| Lignes de CSS | ~580 (design system complet) |
| Bundle JS (gzip) | ~100 KB |
| Build Time | 800ms |
| Dépendances | react, react-dom, react-markdown, lucide-react, vite |
| Compatibilité | Tous navigateurs modernes (Chrome, Firefox, Edge) |
| Mode de déploiement | Single-Server Option A (FastAPI sert le build) |
