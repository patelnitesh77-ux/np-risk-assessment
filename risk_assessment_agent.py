"""
NP Performance Lab - Injury Risk Assessment Agent
18-variable weighted scoring system with traffic light classification
Based on LaLiga research + global literature
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict
import os
from io import BytesIO

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="NP Performance Lab - Risk Assessment", page_icon="🚦", layout="wide")
load_dotenv()

# ============================================================================
# RISK MODEL CONFIGURATION
# ============================================================================

class RiskModelConfig:
    """
    18-Variable Injury Risk Model
    Based on LaLiga research + global literature
    
    Weights: 3 = High impact, 2 = Moderate, 1 = Low
    Adjust these based on your validation data
    """
    
    VARIABLES = {
        # LOAD VARIABLES (Dynamic)
        'acwr_high': {
            'name': 'ACWR > 1.3',
            'weight': 3,
            'threshold': lambda x: x > 1.3,
            'description': 'Acute spike in workload'
        },
        'acwr_low': {
            'name': 'ACWR < 0.8',
            'weight': 2,
            'threshold': lambda x: x < 0.8,
            'description': 'Underload/detraining'
        },
        
        # MATCH EXPOSURE (Dynamic)
        'high_match_exposure': {
            'name': 'High Match Exposure',
            'weight': 2,
            'threshold': lambda mins: mins > 270,  # >3 matches in 7 days
            'description': '>270 mins in last 7 days'
        },
        'low_match_exposure': {
            'name': 'Low Match Exposure',
            'weight': 2,
            'threshold': lambda mins: mins < 45,  # <1 match in 14 days
            'description': '<45 mins in last 14 days'
        },
        
        # COMPETITION FREQUENCY (Dynamic)
        'fixture_congestion': {
            'name': 'Fixture Congestion',
            'weight': 2,
            'threshold': lambda count: count >= 3,  # 3+ matches in 7 days
            'description': '≥3 matches in 7 days'
        },
        
        # GPS VARIABLES (Dynamic)
        'hsr_high': {
            'name': 'High HSR Distance',
            'weight': 2,
            'threshold': lambda x: x > 600,  # >600m per session
            'description': 'HSR >600m per session'
        },
        'sprint_count_high': {
            'name': 'Sprint Count ≥85% Vmax',
            'weight': 2,
            'threshold': lambda x: x > 20,  # >20 sprints
            'description': '>20 sprints ≥85% Vmax'
        },
        'metabolic_power_high': {
            'name': 'High Metabolic Power',
            'weight': 2,
            'threshold': lambda x: x > 25,  # >25 W/kg avg
            'description': 'MPA >25 W/kg'
        },
        'deceleration_high': {
            'name': 'High Deceleration Count',
            'weight': 1,
            'threshold': lambda x: x > 40,  # >40 decelerations
            'description': '>40 high decelerations'
        },
        'max_speed_high': {
            'name': 'Max Speed Spike',
            'weight': 1,
            'threshold': lambda x: x > 32,  # >32 km/h
            'description': 'Max speed >32 km/h'
        },
        
        # RECOVERY VARIABLES (Dynamic)
        'sleep_deficient': {
            'name': 'Sleep Deficiency',
            'weight': 2,
            'threshold': lambda hrs: hrs < 7,
            'description': '<7 hours sleep'
        },
        'wellness_low': {
            'name': 'Low Wellness',
            'weight': 2,
            'threshold': lambda score: score < 65,
            'description': 'Wellness <65/100'
        },
        'soreness_high': {
            'name': 'High Muscle Soreness',
            'weight': 1,
            'threshold': lambda score: score > 6,  # >6/10
            'description': 'Soreness >6/10'
        },
        
        # INJURY HISTORY (Fixed)
        'previous_injury': {
            'name': 'Recent Injury History',
            'weight': 3,
            'threshold': lambda days: days < 60,  # Injured in last 60 days
            'description': 'Injury <60 days ago'
        },
        'recurrent_injury': {
            'name': 'Recurrent Muscle Injury',
            'weight': 3,
            'threshold': lambda count: count >= 2,  # 2+ same-site injuries
            'description': '≥2 injuries same location'
        },
        
        # ENVIRONMENTAL (Fixed)
        'surface_artificial': {
            'name': 'Artificial Surface',
            'weight': 1,
            'threshold': lambda surface: 'artificial' in str(surface).lower() or 'turf' in str(surface).lower(),
            'description': 'Training on artificial turf'
        },
        
        # AGE/MATURATION (Fixed)
        'age_risk': {
            'name': 'Age Risk Factor',
            'weight': 1,
            'threshold': lambda age: age > 28 or age < 17,  # U17 or >28
            'description': 'Age <17 or >28'
        },
        
        # POSITION RISK (Fixed)
        'position_high_risk': {
            'name': 'High-Risk Position',
            'weight': 1,
            'threshold': lambda pos: any(p in str(pos).lower() for p in ['winger', 'forward', 'striker']),
            'description': 'Winger/Forward position'
        },
    }
    
    # Risk classification thresholds
    RISK_THRESHOLDS = {
        'high': 13,      # ≥13 = High risk 🔴
        'moderate': 7,   # 7-12 = Moderate 🟠
        'low': 0         # ≤6 = Low risk 🟢
    }

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class RiskDataProcessor:
    """Process uploaded data for risk assessment"""
    
    @staticmethod
    def calculate_player_scores(player_data: Dict, recent_data: pd.DataFrame) -> Dict:
        """Calculate risk score for a single player"""
        score = 0
        flagged_variables = []
        
        config = RiskModelConfig.VARIABLES
        
        # Evaluate each variable
        for var_id, var_config in config.items():
            try:
                # Get value based on variable type
                value = RiskDataProcessor._get_variable_value(var_id, player_data, recent_data)
                
                if value is not None:
                    # Check if threshold is met
                    is_flagged = var_config['threshold'](value)
                    
                    if is_flagged:
                        score += var_config['weight']
                        flagged_variables.append({
                            'name': var_config['name'],
                            'weight': var_config['weight'],
                            'description': var_config['description'],
                            'value': value
                        })
            except Exception as e:
                continue  # Skip if variable can't be evaluated
        
        # Classify risk level
        if score >= RiskModelConfig.RISK_THRESHOLDS['high']:
            risk_level = 'HIGH'
            color = '🔴'
        elif score >= RiskModelConfig.RISK_THRESHOLDS['moderate']:
            risk_level = 'MODERATE'
            color = '🟠'
        else:
            risk_level = 'LOW'
            color = '🟢'
        
        return {
            'score': score,
            'risk_level': risk_level,
            'color': color,
            'flagged_variables': flagged_variables,
            'total_flags': len(flagged_variables)
        }
    
    @staticmethod
    def _get_variable_value(var_id: str, player_data: Dict, recent_data: pd.DataFrame):
        """Extract variable value from player data"""
        
        # ACWR
        if var_id in ['acwr_high', 'acwr_low']:
            return player_data.get('ACWR', player_data.get('acwr', None))
        
        # Match exposure
        elif var_id == 'high_match_exposure':
            return player_data.get('match_minutes_7d', 0)
        elif var_id == 'low_match_exposure':
            return player_data.get('match_minutes_14d', 0)
        
        # Fixture congestion
        elif var_id == 'fixture_congestion':
            return player_data.get('matches_7d', 0)
        
        # GPS variables
        elif var_id == 'hsr_high':
            return player_data.get('HSR_Distance', player_data.get('hsr', None))
        elif var_id == 'sprint_count_high':
            return player_data.get('Sprint_Count', player_data.get('sprint_count', 0))
        elif var_id == 'metabolic_power_high':
            return player_data.get('Metabolic_Power', player_data.get('mpa', None))
        elif var_id == 'deceleration_high':
            return player_data.get('Deceleration_Count', player_data.get('decel_count', 0))
        elif var_id == 'max_speed_high':
            return player_data.get('Max_Speed', player_data.get('max_speed', None))
        
        # Recovery
        elif var_id == 'sleep_deficient':
            return player_data.get('Sleep_Hours', player_data.get('sleep', None))
        elif var_id == 'wellness_low':
            return player_data.get('Wellness_Score', player_data.get('wellness', None))
        elif var_id == 'soreness_high':
            return player_data.get('Muscle_Soreness', player_data.get('soreness', None))
        
        # Injury history
        elif var_id == 'previous_injury':
            return player_data.get('days_since_injury', 999)
        elif var_id == 'recurrent_injury':
            return player_data.get('injury_count_12m', 0)
        
        # Environmental/Fixed
        elif var_id == 'surface_artificial':
            return player_data.get('surface_type', 'natural')
        elif var_id == 'age_risk':
            return player_data.get('age', 25)
        elif var_id == 'position_high_risk':
            return player_data.get('position', 'midfielder')
        
        return None

# ============================================================================
# LANGGRAPH AGENT
# ============================================================================

class RiskAgentState(TypedDict):
    player_data: pd.DataFrame
    squad_assessment: List[Dict]
    high_risk_players: List[str]
    moderate_risk_players: List[str]
    recommendations: Dict
    logs: List[Dict]

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))

llm = get_llm()

def add_log(state, msg, type="info"):
    if 'logs' not in state:
        state['logs'] = []
    state['logs'].append({'msg': msg, 'type': type})
    return state

def assess_squad_risk(state: RiskAgentState) -> RiskAgentState:
    """Assess risk for entire squad"""
    state = add_log(state, "🚦 Assessing squad injury risk...", "info")
    
    player_data = state['player_data']
    
    if player_data.empty:
        state = add_log(state, "⚠️ No player data", "warning")
        return state
    
    # Get player ID column
    player_col = next((col for col in player_data.columns if 'player' in col.lower() or 'id' in col.lower()), None)
    
    if not player_col:
        state = add_log(state, "⚠️ No player ID column found", "warning")
        return state
    
    players = player_data[player_col].unique()
    
    squad_assessment = []
    high_risk = []
    moderate_risk = []
    
    for player_id in players:
        # Get player's data
        player_rows = player_data[player_data[player_col] == player_id]
        
        # Convert to dict for processing
        player_dict = {}
        for col in player_rows.columns:
            values = player_rows[col].dropna()
            if not values.empty:
                # Use most recent or average depending on column
                if col in ['age', 'position', 'surface_type']:
                    player_dict[col] = values.iloc[0]
                else:
                    player_dict[col] = values.mean() if pd.api.types.is_numeric_dtype(values) else values.iloc[0]
        
        # Calculate risk score
        assessment = RiskDataProcessor.calculate_player_scores(player_dict, player_rows)
        assessment['player_id'] = player_id
        assessment['player_name'] = player_dict.get('Player_Name', player_dict.get('name', player_id))
        
        squad_assessment.append(assessment)
        
        if assessment['risk_level'] == 'HIGH':
            high_risk.append(player_id)
        elif assessment['risk_level'] == 'MODERATE':
            moderate_risk.append(player_id)
    
    # Sort by risk score (highest first)
    squad_assessment.sort(key=lambda x: x['score'], reverse=True)
    
    state['squad_assessment'] = squad_assessment
    state['high_risk_players'] = high_risk
    state['moderate_risk_players'] = moderate_risk
    
    state = add_log(state, f"✓ Assessed {len(players)} players", "success")
    state = add_log(state, f"  🔴 High risk: {len(high_risk)}", "data")
    state = add_log(state, f"  🟠 Moderate risk: {len(moderate_risk)}", "data")
    state = add_log(state, f"  🟢 Low risk: {len(players) - len(high_risk) - len(moderate_risk)}", "data")
    
    return state

def generate_recommendations(state: RiskAgentState) -> RiskAgentState:
    """Generate AI-powered recommendations for high-risk players"""
    state = add_log(state, "💡 Generating recommendations...", "info")
    
    high_risk = state.get('high_risk_players', [])
    squad = state.get('squad_assessment', [])
    
    if not high_risk:
        state['recommendations'] = {'message': 'No high-risk players identified'}
        state = add_log(state, "✓ No interventions needed", "success")
        return state
    
    recommendations = {}
    
    for player_id in high_risk[:5]:  # Top 5 high-risk players
        player_data = next((p for p in squad if p['player_id'] == player_id), None)
        
        if player_data:
            flags = player_data['flagged_variables']
            flag_summary = ", ".join([f"{f['name']} (weight {f['weight']})" for f in flags[:5]])
            
            prompt = f"""Generate 2-3 specific, actionable recommendations for a player at HIGH injury risk.

Player: {player_data['player_name']}
Risk Score: {player_data['score']}
Flagged Variables: {flag_summary}

Provide brief, practical interventions for coaching and medical staff.
Format: Numbered list, 1 sentence each."""

            response = llm.invoke(prompt)
            recommendations[player_id] = {
                'player': player_data['player_name'],
                'score': player_data['score'],
                'recommendations': response.content.strip()
            }
    
    state['recommendations'] = recommendations
    state = add_log(state, f"✓ Generated interventions for {len(recommendations)} players", "success")
    
    return state

def build_risk_graph():
    """Build LangGraph workflow"""
    workflow = StateGraph(RiskAgentState)
    
    workflow.add_node("assess", assess_squad_risk)
    workflow.add_node("recommend", generate_recommendations)
    
    workflow.set_entry_point("assess")
    workflow.add_edge("assess", "recommend")
    workflow.add_edge("recommend", END)
    
    return workflow.compile()

# ============================================================================
# PDF REPORT
# ============================================================================

def generate_risk_pdf(squad_assessment, recommendations, date):
    """Generate PDF risk report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    elements.append(Paragraph("NP Performance Lab", styles['Title']))
    elements.append(Paragraph("Weekly Injury Risk Assessment", styles['Heading2']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary stats
    high = len([p for p in squad_assessment if p['risk_level'] == 'HIGH'])
    mod = len([p for p in squad_assessment if p['risk_level'] == 'MODERATE'])
    low = len([p for p in squad_assessment if p['risk_level'] == 'LOW'])
    
    summary_data = [
        ['Total Players:', str(len(squad_assessment))],
        ['High Risk (≥13):', str(high)],
        ['Moderate Risk (7-12):', str(mod)],
        ['Low Risk (≤6):', str(low)],
    ]
    
    table = Table(summary_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (1,1), (1,1), colors.Color(1, 0.8, 0.8)),
        ('BACKGROUND', (1,2), (1,2), colors.Color(1, 0.9, 0.7)),
        ('BACKGROUND', (1,3), (1,3), colors.Color(0.8, 1, 0.8)),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Squad risk table
    elements.append(Paragraph("Squad Risk Overview", styles['Heading2']))
    
    squad_data = [['Player', 'Score', 'Risk', 'Key Flags']]
    
    # Create paragraph style for wrapping text
    flag_style = ParagraphStyle(
        'FlagStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
    )
    
    for player in squad_assessment[:15]:  # Top 15
        flags = ', '.join([f['name'] for f in player['flagged_variables'][:3]])
        
        # Wrap the flags text in a Paragraph for proper text wrapping
        flags_para = Paragraph(flags if flags else 'None', flag_style)
        
        squad_data.append([
            player['player_name'],
            str(player['score']),
            player['risk_level'],
            flags_para  # Use Paragraph instead of string
        ])
    
    squad_table = Table(squad_data, colWidths=[1.5*inch, 0.6*inch, 0.9*inch, 3.3*inch])
    squad_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),  # Align text to top
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    
    elements.append(squad_table)
    elements.append(PageBreak())
    
    # Recommendations
    if recommendations and 'message' not in recommendations:
        elements.append(Paragraph("Recommended Interventions", styles['Heading2']))
        
        for player_id, rec in recommendations.items():
            elements.append(Paragraph(f"<b>{rec['player']}</b> (Score: {rec['score']})", styles['Heading3']))
            elements.append(Paragraph(rec['recommendations'], styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🚦 NP Performance Lab - Injury Risk Assessment")
st.markdown("**18-Variable Weighted Scoring Model** | Weekly Squad Risk Monitoring")

# Show model info
with st.expander("📚 About the Risk Model"):
    st.markdown("""
    ### How It Works:
    - **18 variables** across load, GPS, recovery, injury history
    - **Weighted scoring**: Each variable carries weight (1-3)
    - **Classification**: High (≥13), Moderate (7-12), Low (≤6)
    
    ### Risk Levels:
    - 🔴 **High Risk (≥13)**: Immediate intervention required
    - 🟠 **Moderate Risk (7-12)**: Close monitoring + modifications
    - 🟢 **Low Risk (≤6)**: Normal training progression
    
    Based on LaLiga research + global literature on muscle injury prevention
    """)

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload Player Data (CSV or Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload file with player load, GPS, recovery, and injury history data"
)

if uploaded_file:
    st.success("✅ File uploaded")
    
    # Load data
    file_ext = uploaded_file.name.split('.')[-1]
    if file_ext == 'csv':
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file)
    else:
        uploaded_file.seek(0)
        data = pd.read_excel(uploaded_file, sheet_name=0)
    
    st.dataframe(data.head(), use_container_width=True)
    
    # Run assessment
    if st.button("▶️ Run Risk Assessment", type="primary", use_container_width=True):
        initial_state = {
            'player_data': data,
            'squad_assessment': [],
            'high_risk_players': [],
            'moderate_risk_players': [],
            'recommendations': {},
            'logs': []
        }
        
        with st.spinner("🤖 Assessing squad risk..."):
            app = build_risk_graph()
            result = app.invoke(initial_state)
        
        st.session_state['risk_result'] = result
        st.session_state['risk_date'] = datetime.now()
        
        # Display results
        st.success("✅ Assessment Complete!")
        
        # Execution log
        with st.expander("📋 Execution Log"):
            for log in result.get('logs', []):
                if log['type'] == 'success':
                    st.success(log['msg'])
                else:
                    st.text(log['msg'])
        
        # Squad overview
        st.subheader("📊 Squad Risk Overview")
        
        squad = result.get('squad_assessment', [])
        high = len([p for p in squad if p['risk_level'] == 'HIGH'])
        mod = len([p for p in squad if p['risk_level'] == 'MODERATE'])
        low = len([p for p in squad if p['risk_level'] == 'LOW'])
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Players", len(squad))
        with cols[1]:
            st.metric("🔴 High Risk", high)
        with cols[2]:
            st.metric("🟠 Moderate Risk", mod)
        with cols[3]:
            st.metric("🟢 Low Risk", low)
        
        # Player cards
        st.markdown("### 🚦 Player Risk Cards")
        
        # Group by risk level
        for risk_level, color in [('HIGH', '🔴'), ('MODERATE', '🟠'), ('LOW', '🟢')]:
            players_at_level = [p for p in squad if p['risk_level'] == risk_level]
            
            if players_at_level:
                st.markdown(f"**{color} {risk_level} RISK** ({len(players_at_level)} players)")
                
                for player in players_at_level[:10]:  # Show top 10 per level
                    with st.expander(f"{player['player_name']} - Score: {player['score']}"):
                        st.write(f"**Flagged Variables ({player['total_flags']}):**")
                        for flag in player['flagged_variables']:
                            st.write(f"- **{flag['name']}** (weight {flag['weight']}): {flag['description']}")
        
        # Recommendations
        if result.get('recommendations') and 'message' not in result['recommendations']:
            st.markdown("### 💡 Recommended Interventions")
            
            for player_id, rec in result['recommendations'].items():
                st.markdown(f"**{rec['player']}** (Score: {rec['score']})")
                st.info(rec['recommendations'])
        
        # PDF Download
        pdf = generate_risk_pdf(squad, result.get('recommendations', {}), st.session_state['risk_date'])
        st.download_button(
            "📄 Download Weekly Risk Report",
            pdf,
            f"risk_assessment_{datetime.now().strftime('%Y%m%d')}.pdf",
            "application/pdf",
            type="primary",
            use_container_width=True
        )

else:
    st.info("👆 Upload player data to begin risk assessment")

st.markdown("---")
st.markdown("Built with LangGraph • NP Performance Lab • 18-Variable Risk Model")
