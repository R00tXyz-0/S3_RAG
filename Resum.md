# Résumé — Pipeline d'ingestion & chunking RAG (Oracle / PL/SQL)

Ce fichier résume le travail réalisé pour la première étape du projet RAG du cours *Bases de Données Avancées*.

> **Mise à jour (étape Visual → Text implémentée)** : le module `src/vision/` ajoute une étape
> **Visual → Text** au pipeline. Chaque page est rendue (PyMuPDF) puis, si elle contient du visuel
> pertinent, envoyée au modèle multimodal `gemini-3.6-flash` (API Google Gemini, SDK `google-genai`) pour
> extraire une description texte de diagrammes/tables/code. Le texte visuel est **fusionné avec la
> couche texte native** avant le nettoyage et le chunking. Les champs `is_ocr` et `DIAGRAM`/`MIXED`
> ont été retirés (plus d'OCR) ; l'équivalent visuel est capturé par `has_visual_content` /
> `visual_processing_status`. Aucun embedding / base vectorielle / retrieval / LLM n'est implémenté.

## Objectif
Construire la couche **ingestion + chunking** d'un système RAG à partir de deux PDF de cours
français, **sans** implémenter (pour l'instant) les embeddings, la base vectorielle, le retrieval,
le reranking ou le LLM.

## PDF traités
- `Cours_Oracle_Complet.pdf` — 133 pages, diaporama (1 slide/page).
- `PLSQL Version Finale (1).pdf` — 155 pages, diaporama également très orienté images.

## Ce qui a été fait

### 1. Extraction
- `src/extraction/extractor.py` : récupère la couche texte et le nombre d'images intégrées par page
  (pypdf, pur Python).

### 2. Classification des pages (sans plages codées en dur)
- `src/classification/page_classifier.py` : `native_text` / `hybrid` (titre seul + corps image) /
  `image_only`, décidée **uniquement** à partir de la qualité du texte extrait.
- Oracle : 69 native, 23 hybrid, 41 image-only. PL/SQL : 153 hybrid (deck quasi entièrement image).

### 3. Visual → Text — **implémenté avec Gemini (`src/vision/`)**
- `src/vision/renderer.py` : rendu de chaque page PDF en image via PyMuPDF (import différé).
- `src/vision/detector.py` : ne traite que les pages `no_text` / `hybrid` et les pages natives
  contenant une grande image (un diagramme) ; ignore les icônes décoratives. Déduplication des
  pages rendues identiques.
- `src/vision/model.py` : wrapper `GeminiVisionModel` basé sur l'API Google Gemini (`google-genai`,
  modèle `gemini-3.6-flash` par défaut). Clé lue depuis `GEMINI_API_KEY` (`.env` via python-dotenv),
  jamais codée en dur ni affichée. Timeout, retry avec backoff exponentiel, gestion 429, logging
  par page, mockable pour les tests.
- `src/vision/merger.py` + `vision/stage.py` : orchestration par page, cache des résultats réussis
  (`data/processed/vision/<source>/page_NNN.json`), fusion texte natif + visuel avant nettoyage.
  En cas d'échec (rendu/API), le texte natif est préservé et l'erreur est enregistrée dans
  `visual_processing_status`.

### 4. Nettoyage
- `src/cleaning/cleaner.py` : suppression des en-têtes/pieds de page récurrents par **fréquence**
  (ligne présente sur ≥ 25 % des pages) + regex configurables (email, signature prof., "Année
  universitaire"). Les termes techniques (Oracle, SGA, CREATE, …) sont préservés.

### 5. Détection de structure (générique / adaptative)
- `src/structure/detector.py` : détecte les chapitres (regex `Chapitre`/`Partie`) et groupe les pages
  consécutives partageant un titre (proche-)identique. Fonctionne pour diaporamas et pour documents
  type livre. Le vrai titre de slide est trouvé en ignorant les « labels de deck » répétés.
- Oracle : **Chapitre 1** (1–37) et **Chapitre 3** (38–133) détectés — **aucun Chapitre 2 inventé**.
- PL/SQL : pas de marqueur `Chapitre` → le titre du deck « PL/SQL » sert de chapitre.

### 6. Chunking structure-aware
- `src/chunking/chunker.py` : hiérarchie Chapitre → groupe de titres → page → contenu.
  - Fusion des petites pages d'un **même groupe** ; un groupe n'est divisé qu'au-delà de la limite.
  - Cible ≈ 550 tokens, overlap ≈ 64, max ≈ 800, min préféré ≈ 150.
  - **Jamais** de coupure au milieu d'une instruction SQL (blocs de code atomiques).
  - **Jamais** d'overlap ni de fusion à travers une frontière de chapitre / de groupe de titres.

### 7. Métadonnées
- `src/metadata/metadata.py` : `source, page_start, page_end, chapter, slide_title, content_type,
  has_code, chunk_index` (+ `token_count`, `pages`). Champs visuels ajoutés quand pertinents :
  `has_visual_content, vision_model, visual_processing_status`. (Le champ `is_ocr` a été retiré ;
  l'équivalent visuel est capturé par `has_visual_content` / `visual_processing_status`.) **Pas** de
  champ `subsection` fabriqué.

### 8. Validation
- `src/validation/validator.py` : continuité des pages, statut d'extraction, chunks non vides,
  métadonnées valides, plages de pages, `chunk_index` unique, intégrité du code. Les chunks < 150
  tokens sont **signalés pour relecture, jamais supprimés**.

### 9. Orchestration & rapport
- `src/ingestion/pdf_loader.py` + `report.py` + `run_ingestion.py` : pipeline complet et rapport
  (pages, types, chapitres, groupes, chunks, min/max/avg tokens, code, diagrammes, OCR, validation).
- Sorties : `data/processed/evaluation/report_*.json` et `data/processed/chunks/chunks_*.jsonl`.

## Résultats obtenus
| PDF | Pages | Chapitres | Groupes | Chunks | Erreurs validation | Couverture |
|-----|-------|-----------|---------|--------|--------------------|------------|
| Oracle | 133 | 2 | 68 | 69 | 0 | 1–133 ✓ (49–76, 120–133 présents) |
| PL/SQL | 155 | 1 | 127 | 127 | 0 | 1–155 ✓ |

## Tests
- `tests/unit/` (chunker, classifier, cleaner, detector, metadata, validator, vision) +
  `tests/integration/` (pipeline réel sur les deux PDF, y compris avec un stage visuel mocké) →
  **30 tests passants**.
- Vérifie notamment : pages 49–76 / 120–133 représentées, absence de Chapitre 2, groupes de titres
  consécutifs, protection du code SQL, classification image-only, fusion des petits chunks.

## Limites connues
- L'étape Visual → Text nécessite `google-genai`, `python-dotenv`, `pymupdf` et `Pillow`. S'ils ne sont
  pas installés, ou si la clé Gemini est absente / la requête échoue, le pipeline retombe
  automatiquement sur le mode textuel : le texte natif est préservé et les pages `no_text` ne
  produisent pas de chunk.
- Beaucoup de chunks < 150 tokens (propre aux diaporamas, ~60 tokens/slide) → signalés, pas cachés.
- Comptage de tokens = approximation par nombre de mots (suffisant avant embedding).
- L'inférence Gemini est volontairement mockée dans les tests (un `generate_fn` est injecté) ;
  exécuter le vrai modèle ne nécessite qu'une clé `GEMINI_API_KEY` (aucun GPU / téléchargement de
  modèle local).

## Hors périmètre (étapes suivantes)
Embeddings → Base vectorielle → Retrieval → Reranking → LLM → API → UI. Non implémentés.

## Prochaine étape recommandée
Installer les dépendances vision (`pip install -r requirements.txt`), relancer
`python run_ingestion.py` pour produire les chunks enrichis par le stage Visual → Text, puis
démarrer l'étape **embeddings + base vectorielle**.
