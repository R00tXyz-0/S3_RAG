# Résumé — Pipeline d'ingestion & chunking RAG (Oracle / PL/SQL)

Ce fichier résume le travail réalisé pour la première étape du projet RAG du cours *Bases de Données Avancées*.

> **Pipeline natif (texte uniquement)** : l'étape *Visual → Text* (Gemini/OCR/rendu d'images) a été
> supprimée. Le pipeline utilise uniquement la couche texte native extraite des PDF. Aucun embedding /
> base vectorielle / retrieval / LLM n'est implémenté.

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

### 3. Nettoyage
- `src/cleaning/cleaner.py` : suppression des en-têtes/pieds de page récurrents par **fréquence**
  (ligne présente sur ≥ 25 % des pages) + regex configurables (email, signature prof., "Année
  universitaire"). Les termes techniques (Oracle, SGA, CREATE, …) sont préservés.

### 4. Détection de structure (générique / adaptative)
- `src/structure/detector.py` : détecte les chapitres (regex `Chapitre`/`Partie`) et groupe les pages
  consécutives partageant un titre (proche-)identique. Fonctionne pour diaporamas et pour documents
  type livre. Le vrai titre de slide est trouvé en ignorant les « labels de deck » répétés.
- Oracle : **Chapitre 1** (1–37) et **Chapitre 3** (38–133) détectés — **aucun Chapitre 2 inventé**.
- PL/SQL : pas de marqueur `Chapitre` → le titre du deck « PL/SQL » sert de chapitre.

### 5. Chunking structure-aware
- `src/chunking/chunker.py` : hiérarchie Chapitre → groupe de titres → page → contenu.
  - Fusion des petites pages d'un **même groupe** ; un groupe n'est divisé qu'au-delà de la limite.
  - Cible ≈ 550 tokens, overlap ≈ 64, max ≈ 800, min préféré ≈ 150.
  - **Jamais** de coupure au milieu d'une instruction SQL (blocs de code atomiques).
  - **Jamais** d'overlap ni de fusion à travers une frontière de chapitre / de groupe de titres.

### 6. Métadonnées
- `src/metadata/metadata.py` : `source, page_start, page_end, chapter, slide_title, content_type,
  has_code, chunk_index` (+ `token_count`, `pages`). **Pas** de champ `subsection` fabriqué.

### 7. Validation
- `src/validation/validator.py` : continuité des pages, statut d'extraction, chunks non vides,
  métadonnées valides, plages de pages, `chunk_index` unique, intégrité du code. Les chunks < 150
  tokens sont **signalés pour relecture, jamais supprimés**.

### 8. Orchestration & rapport
- `src/ingestion/pdf_loader.py` + `report.py` + `run_ingestion.py` : pipeline complet et rapport
  (pages, types, chapitres, groupes, chunks, min/max/avg tokens, code, validation).
- Sorties : `data/processed/evaluation/report_*.json` et `data/processed/chunks/chunks_*.jsonl`.

## Résultats obtenus
| PDF | Pages | Chapitres | Groupes | Chunks | Erreurs validation | Couverture |
|-----|-------|-----------|---------|--------|--------------------|------------|
| Oracle | 133 | 2 | 68 | 69 | 0 | 1–133 ✓ (49–76, 120–133 présents) |
| PL/SQL | 155 | 1 | 127 | 127 | 0 | 1–155 ✓ |

## Tests
- `tests/unit/` (chunker, classifier, cleaner, detector, metadata, validator) +
  `tests/integration/` (pipeline réel sur les deux PDF) → **tests passants**.
- Vérifie notamment : pages 49–76 / 120–133 représentées, absence de Chapitre 2, groupes de titres
  consécutifs, protection du code SQL, classification image-only, fusion des petits chunks.

## Limites connues
- Beaucoup de chunks < 150 tokens (propre aux diaporamas, ~60 tokens/slide) → signalés, pas cachés.
- Comptage de tokens = approximation par nombre de mots (suffisant avant embedding).
- Les pages `no_text` / `hybrid` (sans couche texte utilisable) ne produisent pas de chunk ;
  seul le texte natif est utilisé.

## Hors périmètre (étapes suivantes)
Embeddings → Base vectorielle → Retrieval → Reranking → LLM → API → UI. Non implémentés.

## Prochaine étape recommandée
Relancer `python run_ingestion.py` pour produire les chunks, puis démarrer l'étape
**embeddings + base vectorielle**.
