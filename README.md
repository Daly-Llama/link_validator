# **Link Validator — End‑to‑End Machine Learning Pipeline**

A complete, modular machine‑learning system that predicts whether a Help Desk analyst attached the correct knowledge article to a support ticket.  
Designed for organizations practicing **Knowledge‑Centered Service (KCS)** and built as part of DSC 680 – Applied Data Science at Bellevue University.

This project demonstrates:

- Web scraping  
- HTML parsing  
- Synthetic data generation using OpenAI  
- Text embedding with SentenceTransformers  
- Cosine similarity search  
- Supervised learning with Logistic Regression  
- Model evaluation and explainability (ROC, PR, SHAP)  
- A fully reproducible, script‑by‑script ML pipeline  

---

# **Why This Project Matters for Organizations that practice KCS**

Organizations practicing **KCS** depend on analysts consistently linking incidents to the correct knowledge articles. These links drive the core KCS outcomes: findability, reuse, content health, and continuous improvement. But in real support environments, analysts often mislink articles due to time pressure, vague incident descriptions, or unfamiliarity with the knowledge base.

This project helps solve that problem.

A Link Validator model provides:

- **Real‑time feedback** that improves analyst accuracy  
- **Cleaner reuse data**, which strengthens both the Solve and Evolve Loops  
- **Better content governance**, by surfacing articles that are frequently mislinked  
- **Reduced cognitive load**, especially for new analysts  
- **Scalable quality control**, replacing manual link audits with automated insight  

By combining embeddings, similarity search, and an interpretable classifier, this system enhances the reliability of KCS processes and supports a healthier, more effective knowledge ecosystem.

---

# **Quickstart**

If you want to run the entire pipeline end‑to‑end, here’s the fastest way to get started.

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/link_validator.git
cd link_validator
```

### **2. Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Add your OpenAI API key**
Create a file named `OPENAI.json`:

```json
{
  "openai": "YOUR_API_KEY_HERE"
}
```

### **5. Run the pipeline**
```bash
python kb_scraper.py
python html_parser.py
python incident_generator.py
python text_embedder.py
python link_generator.py
python text_classifier.py
python model_evaluator.py
python model_explainer.py   # optional
```

That’s it — you now have:

- Parsed article JSON  
- Synthetic incidents  
- Embeddings  
- A labeled link dataset  
- A trained model  
- Evaluation plots  
- SHAP explanations  

---

# **Project Overview**

Help Desk analysts often struggle to find the correct knowledge article for a customer issue.  
This project builds a **Link Validator** model that predicts whether an incident is linked to the correct article.

The pipeline:

1. Scrapes public knowledge articles  
2. Parses and structures article content  
3. Generates synthetic incidents using OpenAI  
4. Embeds articles and incidents  
5. Computes cosine similarity to create link candidates  
6. Trains a Logistic Regression classifier  
7. Evaluates and explains the model  

The result is a lightweight, interpretable model suitable for real‑time use in support environments.

---

# **Repository Structure**

```
link_validator/
│
├── embeddings/                 # Article + incident embeddings (JSONL)
│   ├── articles.jsonl
│   └── incidents.jsonl
│
├── prompts/                    # Prompt template + situational modifiers
│   ├── incident_prompt.txt
│   └── situational_modifiers/
│
├── raw_data/                   # Scraped + parsed data
│   ├── html/                   # Raw HTML (not included due to restrictions)
│   ├── json/                   # Parsed article JSON
│   ├── inc_json/               # Synthetic incident JSON
│   └── scrape_log.csv
│
├── kb_scraper.py               # Step 1: Scrape KB articles
├── html_parser.py              # Step 2: Parse HTML → JSON
├── incident_generator.py       # Step 3: Generate synthetic incidents
├── text_embedder.py            # Step 4: Create embeddings
├── link_generator.py           # Step 5: Build link dataset
├── text_classifier.py          # Step 6: Train the model
├── model_evaluator.py          # Step 7: Evaluate the model
├── model_explainer.py          # Step 8: SHAP explainability
│
├── links.jsonl                 # Final link dataset
├── model.pkl                   # Trained Logistic Regression model
├── OPENAI.json                 # Local API key file (ignored by Git)
└── .gitignore
```
---

# **How to Customize Prompts & Situational Modifiers**

The incident generation approach uses a flexible prompt‑engineering system that allows users to control:

- Writing style  
- Grammar quality  
- Technical proficiency  
- Noise level  
- Structure  
- Detail level  
- Entry method  
- Native language  

This section explains how to modify or extend that system.

---

## **1. Prompt Template (`incident_prompt.txt`)**

This file contains the base prompt used to generate synthetic incidents.

It includes two placeholders:

- `[ARTICLE_TEXT]`  
- `[SITUATIONAL_MODIFIERS]`  

You can freely modify:

- Instructions  
- Tone  
- Output schema  
- Formatting  

As long as the placeholders remain, the pipeline will work.

---

## **2. Situational Modifiers (`prompts/situational_modifiers/*.json`)**

Each JSON file contains weighted options for a specific dimension of variability.

Example: detail_level.json

```json
{
  "version": 1,
  "items": {
    "1": { "name": "vague", "value": "very limited detail", "weight": 0.30 },
    "2": { "name": "moderately_detailed", "value": "reasonable amount of detail", "weight": 0.50 },
    "3": { "name": "overly_detailed", "value": "excessive unnecessary detail", "weight": 0.20 }
  }
}

```

The script:

- Randomly selects one option per file  
- Combines them into a modifier paragraph  
- Injects them into the prompt  

This creates realistic variability in incident text.

---

## **3. Adding New Modifier Categories**

To add a new dimension:

1. Create a new JSON file in `prompts/situational_modifiers/`
2. Follow the same structure (`items → key → weight/value`)
3. Add a new line in `build_situational_modifier()` within the `incident_generator.py` file
4. Append the value to the final modifier paragraph

The system is intentionally modular and easy to extend.

---

## **4. Adjusting Randomness**

You can control variability by adjusting:

- Weights  
- Number of modifiers  
- OpenAI temperature (currently 0.0 for determinism)  

For more chaotic incidents, increase temperature to ~0.3–0.5.

---

# **Model Performance (Summary)**

Without modifications, this model achieves:

- **Accuracy:** 0.9041  
- **Precision:** 0.7801  
- **Recall:** 0.7182  
- **F1 Score:** 0.7479  
- **ROC‑AUC:** 0.9467  

These results indicate a well‑balanced model that:

- Is highly precise when predicting correct links  
- Maintains strong recall  
- Performs consistently across thresholds  

---

# **Interpretability**

Using SHAP, the most influential features were:

1. **Distance (1 – cosine similarity)**  
2. **Cosine similarity**  
3. **Subcategory match**  

This aligns with expectations: semantic similarity and metadata alignment drive correct linking.

---

# **Data Availability & Restrictions**

Due to University of South Dakota policy:

- Raw HTML articles **cannot be redistributed**  
- Parsed article JSON is also excluded  

Users must scrape their own KB articles or adapt the pipeline to their environment.

Synthetic incidents, embeddings, and link data **are included**.

---

# **Future Improvements**

- Use a more powerful OpenAI model for richer noise simulation  
- Add a threshold slider to the UI for analysts  
- Incorporate real incident samples into prompt generation  
- Expand feature engineering with TF‑IDF or cross‑encoders  

---

# **Conclusion**

This project demonstrates a complete, reproducible ML pipeline for validating knowledge article links in a KCS environment.  
It balances:

- Practicality  
- Interpretability  
- Real‑world applicability
