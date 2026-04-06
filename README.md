# 🚦 NP Performance Lab - Injury Risk Assessment Agent

**AI-powered weekly risk screening for football squads**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://np-risk-assessment.streamlit.app/)

---

## 🎯 What It Does

Weekly injury risk assessment for entire squads using 18-variable weighted scoring:

- **Upload** squad data (load, GPS, recovery, injury history)
- **Scores** each player across 18 research-backed variables
- **Classifies** risk level: 🔴 High / 🟠 Moderate / 🟢 Low
- **Generates** weekly PDF reports with intervention recommendations

**Replaces manual risk screening** with AI-powered automation

---

## ✨ Features

### 🧮 18-Variable Scoring Model

Based on LaLiga research + global injury prevention literature:

**Load Variables (weight 2-3):**
- ACWR >1.3 or <0.8
- Match exposure (high/low)
- Fixture congestion

**GPS Metrics (weight 1-2):**
- High-speed running
- Sprint count ≥85% Vmax
- Metabolic power
- Deceleration count
- Max speed spikes

**Recovery (weight 1-2):**
- Sleep deficiency
- Low wellness score
- High muscle soreness

**Injury History (weight 3):**
- Recent injury <60 days
- Recurrent injuries

**Fixed Factors (weight 1):**
- Artificial surface
- Age risk (<17 or >28)
- High-risk position

### 🚦 Traffic Light Classification

- 🔴 **High Risk (≥13)**: Immediate intervention required
- 🟠 **Moderate Risk (7-12)**: Close monitoring + load modifications
- 🟢 **Low Risk (≤6)**: Normal training progression

### 🤖 AI Recommendations

- Top 5 high-risk players get personalized interventions
- AI analyzes flagged variables → specific actions
- Practical guidance for coaches, physios, S&C staff

---

## 🚀 Quick Start

### 1. Get Your Groq API Key (Free)

This agent uses **Groq** for AI-powered recommendations:

1. Go to **https://console.groq.com**
2. Sign up (free account)
3. Navigate to **API Keys**
4. Click **"Create API Key"**
5. Copy your key (starts with `gsk_...`)

**Why Groq?**
- ✅ Free tier: 14,400 requests/day
- ✅ Llama 3.3-70B model
- ✅ Fast inference (<1 second)
- ✅ No credit card required
- ✅ Production-ready

### 2. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/np-risk-assessment.git
cd np-risk-assessment

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_actual_key_here" > .env
```

### 3. Run the Agent

```bash
streamlit run risk_assessment_agent.py
```

Open **http://localhost:8501** in your browser

### 4. Upload Squad Data

**Sample data included:** `squad_risk_data.csv` (20 players)

Or upload your own squad data.

---

## 📁 Data Format

### Required Columns

```csv
Player_ID, Player_Name, Age, Position,
ACWR, Sleep_Hours, Wellness_Score, 
HSR_Distance, Sprint_Count, Metabolic_Power,
Deceleration_Count, Max_Speed, Muscle_Soreness,
match_minutes_7d, match_minutes_14d, matches_7d,
days_since_injury, injury_count_12m, surface_type
```

**Column names flexible** - agent matches patterns

### Example Row

```csv
P001, Player 1, 22, Midfielder,
1.35, 6.2, 58,
720, 18, 26.5,
42, 31.2, 7,
270, 90, 3,
25, 1, Natural
```

### Optional Columns

- Player photo URL
- Contract expiry
- Medical notes
- Custom risk factors

---

## 🧠 How It Works

### LangGraph Workflow

```
User Upload
    ↓
assess_squad_risk
    ├─→ Calculate score per player (18 variables)
    ├─→ Apply weights (1-3 per variable)
    ├─→ Classify: High / Moderate / Low
    └─→ Identify high-risk players
    ↓
generate_recommendations
    ├─→ AI analyzes top 5 high-risk players
    ├─→ Reviews flagged variables
    └─→ Creates specific interventions
    ↓
PDF Report
```

### Scoring Logic

```python
# Each variable checked against threshold
for variable in 18_VARIABLES:
    if player_data meets threshold:
        score += variable.weight  # 1, 2, or 3
        
# Final classification
if score >= 13:
    risk = HIGH (🔴)
elif score >= 7:
    risk = MODERATE (🟠)
else:
    risk = LOW (🟢)
```

---

## 🛠️ Tech Stack

- **LangGraph** - Multi-agent orchestration
- **Groq** - Free AI inference (Llama 3.3-70B)
- **Streamlit** - Web interface
- **ReportLab** - PDF generation
- **Pandas/NumPy** - Data processing

---

## 📊 Sample Output

### Squad Overview

```
Total Players: 20
🔴 High Risk: 4
🟠 Moderate Risk: 6
🟢 Low Risk: 10
```

### Individual Risk Cards

```
Player 3 - Score: 23 (HIGH)
Flagged Variables (8):
- ACWR > 1.3 (weight 3)
- High Match Exposure (weight 2)
- Fixture Congestion (weight 2)
- Sleep Deficiency (weight 2)
- Recent Injury <60d (weight 3)
...
```

### AI Recommendations

```
Player 3 (Score: 23)
1. Reduce training load by 20% for next 7 days
2. Implement additional recovery session (sleep hygiene)
3. Monitor for early signs of fatigue/injury
```

---

## 🔧 Configuration

### Adjust Risk Thresholds

Edit in `risk_assessment_agent.py`:

```python
class RiskModelConfig:
    RISK_THRESHOLDS = {
        'high': 13,      # Change to 15 for stricter classification
        'moderate': 7,   # Change to 9
        'low': 0
    }
```

### Customize Variable Weights

```python
VARIABLES = {
    'acwr_high': {
        'weight': 3,  # Change to 4 for more impact
        'threshold': lambda x: x > 1.3,  # Adjust threshold
    },
    # ... other variables
}
```

### Add Custom Variables

```python
'custom_metric': {
    'name': 'Custom Risk Factor',
    'weight': 2,
    'threshold': lambda x: x > threshold_value,
    'description': 'Your description'
}
```

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "API key not found"
- Check `.env` file: `GROQ_API_KEY=gsk_...`
- Restart Streamlit after creating .env

### PDF text overflow
- Already fixed in latest version
- Update `risk_assessment_agent.py`

### "No high-risk players" but expected some
- Check variable thresholds
- Review player data quality
- Adjust weights if needed

---

## 📈 Use Cases

### Weekly Squad Screening
- Upload Monday morning data
- Identify high-risk players before midweek fixture
- Adjust training loads accordingly

### Pre-Season Risk Audit
- Assess all 90 academy players
- Prioritize who needs extra monitoring
- Plan individualized programs

### Return-to-Play Validation
- Check player's risk score post-injury
- Validate readiness for competition
- Evidence-based clearance decisions

### Monthly Reporting
- Track risk trends over time
- Report to academy director
- Justify resource allocation

---

## 🔬 Research Foundation

**Based on peer-reviewed literature:**

1. **LaLiga Injury Risk Model** - 18-variable framework
2. **Gabbett TJ (2016)** - ACWR and injury risk
3. **Bengtsson H et al. (2013)** - Fixture congestion impact
4. **Moreno-Pérez V et al. (2022-2023)** - Match exposure thresholds
5. **Laux P et al. (2015)** - Recovery-stress balance

**Validation:**
- 30% reduction in muscular injuries (U17 squad, 4 months)
- Weekly screening time: Manual 2hrs → Automated 2mins

---

## 🤝 Contributing

Portfolio/research project. Feedback welcome!

**Customize for your academy:**
- Adjust thresholds based on your data
- Add academy-specific variables
- Validate against injury outcomes

**Contact:** [Nitesh Patel]

---

## 📄 License

MIT License - Free for personal and commercial use

---

## 🙏 Credits

Built by **Nitesh Patel** (www.linkedin.com/in/niteshppatel)

FIFA Diploma in Football Medicine | ASCA Level 1

**Research:**
- LaLiga Medical Department
- UEFA Injury Study Group
- Global injury prevention literature

**Tech Stack:**
- [Groq](https://groq.com) - Free AI inference
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Streamlit](https://streamlit.io)

---

## 🚀 Live Demo

Try it now: **[[YOUR_DEPLOYED_URL](https://np-risk-assessment.streamlit.app/)L]**

Sample squad data included (20 players)

---

## 📞 For Football Academies

**Want this system for your academy?**

- Customized for your data structure
- On-premise or cloud deployment
- Training for coaching/medical staff
- Ongoing support and updates

Contact: [(patelnitesh77@gmail.com) / [www.linkedin.com/in/niteshppatel]]

---

**Built with LangGraph • Powered by Groq • NP Performance Lab**
