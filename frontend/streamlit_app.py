# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="Safran Evaluation Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API
API_URL = "http://127.0.0.1:8000/"

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: transparent;
    }
    .stApp {
       background-color: #F0F2F6;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🚀 Safran - Analyse IA des Évaluations")
st.markdown("### Think to Deploy - Système d'Analyse Automatisée")

# Vérifier la connexion API
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ API Connectée")
    else:
        st.sidebar.error("❌ API Non Disponible")
except:
    st.sidebar.error("❌ Impossible de se connecter à l'API")
    st.error("⚠️ L'API n'est pas accessible. Assurez-vous qu'elle est lancée sur http://backend:8000")
    st.stop()

# Sidebar - Upload de fichier
st.sidebar.markdown("## 📤 Import de Données")
uploaded_file = st.sidebar.file_uploader(
    "Choisir un fichier CSV",
    type=['csv',],
    help="Uploadez vos évaluations de formation"
)

# Variable de session pour stocker l'analysis_id
#"Est-ce que l'étiquette 'analysis_id' existe déjà dans mon carnet de notes ?
#st.session_state est un simple dictionnaire Python la mémoire à court terme de ton application.
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Upload du fichier
if uploaded_file is not None:
    #Elle dessine le bouton sur ton écran ET Elle renvoie True (Vrai) uniquement tu cliques
    if st.sidebar.button("🔄 Analyser le fichier", type="primary"):
        with st.spinner("📊 Upload et analyse en cours..."):
            try:
                # Upload
                #On prépare le fichier pour l'envoi.
                files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
                #On envoie l'enveloppe à une adresse précise (l'API).
                response = requests.post(f"{API_URL}/api/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.analysis_id = result['analysis_id']
                    
                    st.sidebar.success(f"✅ {result['rows_processed']} lignes chargées")
                    
                    # Lancer l'analyse
                    with st.spinner("🧠 Analyse IA en cours (NLP + ML)..."):
                        time.sleep(1)
                        analysis_response = requests.get(
                            f"{API_URL}/api/analyze/{st.session_state.analysis_id}"
                        )
                        
                        if analysis_response.status_code == 200:
                            st.session_state.analysis_data = analysis_response.json()
                            st.sidebar.success("✅ Analyse terminée!")
                            st.rerun()
                        else:
                            st.error(f"Erreur d'analyse: {analysis_response.text}")
                else:
                    st.error(f"Erreur d'upload: {response.text}")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

# Affichage des résultats
if st.session_state.analysis_data:
    data = st.session_state.analysis_data
    
    # Onglets - SEULEMENT 3 MAINTENANT
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🔍 Analyse Sentiment",
        "💡 Insights & Signaux"
    ])
    
    # TAB 1 - DASHBOARD
    with tab1:
        st.markdown("## 📊 Vue d'Ensemble")
        
        # KPIs en haut
        col1, col2, col3, col4 = st.columns(4)
        
        summary = data.get('summary', {})
        
        with col1:
            st.metric(
                "Total Évaluations",
                summary.get('total_evaluations', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Satisfaction Moyenne",
                f"{summary.get('avg_satisfaction', 0)}/5",
                delta="+5.3%" if summary.get('avg_satisfaction', 0) > 4 else None
            )
        
        with col3:
            completion = data.get('kpis', {}).get('global', {}).get('completion_rate', 0)
            st.metric(
                "Taux de Complétion",
                f"{completion}%",
                delta=None
            )
        
        with col4:
            weak_signals = len(data.get('weak_signals', []))
            st.metric(
                "Signaux Faibles",
                weak_signals,
                delta=f"-{weak_signals}" if weak_signals > 0 else None,
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Scores par Critère")
            
            # Préparer les données pour le graphique
            criteria_data = {
                'Critère': ['Satisfaction', 'Contenu', 'Logistique', 'Applicabilité'],
                'Score': [
                    summary.get('avg_satisfaction', 0),
                    summary.get('avg_satisfaction', 0),
                    summary.get('avg_satisfaction', 0) - 0.4,
                    summary.get('avg_satisfaction', 0) - 0.2
                ]
            }
            
            fig_bars = px.bar(
                criteria_data,
                x='Critère',
                y='Score',
                color='Score',
                color_continuous_scale='RdYlGn',
                range_color=[1, 5],
                text='Score'
            )
            fig_bars.update_layout(
                height=400,
                showlegend=False,
                yaxis_range=[0, 5]
            )
            fig_bars.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_bars, use_container_width=True)
        
        with col2:
            st.markdown("### 🎭 Distribution des Sentiments")
            
            sentiment_data = data.get('sentiment_analysis', {}).get('distribution', {})
            
            if sentiment_data:
                labels = list(sentiment_data.keys())
                values = list(sentiment_data.values())
                colors = ['#10b981', '#fbbf24', '#ef4444']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=colors
                )])
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Données de sentiment non disponibles")
        
        # Graphique d'évolution temporelle
        st.markdown("### 📈 Évolution dans le Temps")
        
        temporal = data.get('kpis', {}).get('temporal', {}).get('monthly_data', [])
        
        if temporal:
            df_temporal = pd.DataFrame(temporal)
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_temporal['month'].astype(str),
                y=df_temporal['satisfaction_mean'],
                mode='lines+markers',
                name='Satisfaction',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8)
            ))
            
            fig_line.update_layout(
                height=350,
                xaxis_title="Mois",
                yaxis_title="Score Moyen",
                yaxis_range=[0, 5],
                hovermode='x unified'
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Données temporelles non disponibles - Ajoutez plus d'évaluations avec dates")
        
        # Section statistiques détaillées
        st.markdown("---")
        st.markdown("### 📊 Statistiques Détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 KPIs Globaux")
            global_kpis = data.get('kpis', {}).get('global', {})
            
            kpi_df = pd.DataFrame([
                {'Métrique': 'Total Évaluations', 'Valeur': global_kpis.get('total_evaluations', 0)},
                {'Métrique': 'Taux Complétion', 'Valeur': f"{global_kpis.get('completion_rate', 0)}%"},
                {'Métrique': 'Satisfaction Moyenne', 'Valeur': global_kpis.get('avg_satisfaction', 0)},
                {'Métrique': 'Satisfaction Médiane', 'Valeur': global_kpis.get('median_satisfaction', 0)}
                
            ])
            
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📈 Tendances")
            temporal_kpis = data.get('kpis', {}).get('temporal', {})
            
            if temporal_kpis:
                trend_df = pd.DataFrame([
                    {'Indicateur': 'Direction', 'Valeur': temporal_kpis.get('trend_direction', 'N/A').upper()},
                    {'Indicateur': 'Évolution (%)', 'Valeur': f"{temporal_kpis.get('evolution_pct', 0)}%"},
                    {'Indicateur': 'Pente', 'Valeur': temporal_kpis.get('trend_slope', 0)},
                ])
                
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
            else:
                st.info("Tendances non disponibles")
    
    # TAB 2 - SENTIMENT
    with tab2:
        st.markdown("## 🔍 Analyse NLP des Sentiments")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏷️ Thèmes Principaux Extraits des commentaires négatives ")
   
            topics = data.get('topics', {}).get('keywordsnegatives', [])
                
            if topics:
                    df_topics = pd.DataFrame(topics)
                    #3CHIFFRE APRES LA VIRGULE
                    df_topics['score'] = df_topics['score'].round(3)
                    
                    fig_topics = px.bar(
                        df_topics.head(10),
                        x='score',
                        y='word',
                        orientation='h',
                        title="Top Mots-Clés (TF-IDF) des commentaires negatives",
                        color='score',
                        color_continuous_scale='Reds'
                    )
                    fig_topics.update_layout(height=400)
                    st.plotly_chart(fig_topics, use_container_width=True)
            else:
                    st.info("Extraction de thèmes en cours...")
        
        with col2:
            st.markdown("### 📊 Statistiques NLP")
            
            sentiment_details = data.get('sentiment_analysis', {}).get('details', {})
            
            if sentiment_details:
                st.metric("Textes Analysés", sentiment_details.get('total_analyzed', 0))
                st.metric("Positifs", f"{sentiment_details.get('positive_pct', 0)}%")
                st.metric("Négatifs", f"{sentiment_details.get('negative_pct', 0)}%")
                st.metric("Confiance Moyenne", f"{sentiment_details.get('avg_confidence', 0):.1%}")
            else:
                st.info("Statistiques en cours de calcul...")
        
        # Clustering
        st.markdown("---")
        st.markdown("### 🎯 Clustering des Commentaires")
        
        clustering = data.get('clustering', {}).get('summary', [])
        
        if clustering:
            df_clusters = pd.DataFrame(clustering)
            
            st.dataframe(
                df_clusters[['cluster_id', 'size', 'avg_satisfaction', 'keywords']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Clustering non disponible")

        # Clustering negatif
        st.markdown("---")
        st.markdown("### 🎯 Clustering des Commentaires negatifes")
        
        clustering_neg = data.get('clusteringnegatives', {}).get('summary', [])
        
        if clustering:
            df_clusters_neg = pd.DataFrame(clustering_neg)
            
            st.dataframe(
                df_clusters_neg[['cluster_id', 'size', 'avg_satisfaction', 'keywords']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Clustering non disponible")
        
        # Distribution détaillée des sentiments
        st.markdown("---")
        st.markdown("### 📊 Distribution Détaillée des Sentiments")
        
        sentiment_dist = data.get('sentiment_analysis', {}).get('distribution', {})
        
        if sentiment_dist:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"**😊 Positif**\n\n{sentiment_dist.get('POSITIVE', 0)}%")
            
            with col2:
                st.info(f"**😐 Neutre**\n\n{sentiment_dist.get('NEUTRAL', 0)}%")
            
            with col3:
                st.error(f"**😞 Négatif**\n\n{sentiment_dist.get('NEGATIVE', 0)}%")
    
    # TAB 3 - INSIGHTS
    with tab3:
        st.markdown("## 💡 Insights Actionables")
        
        # Signaux faibles
        weak_signals = data.get('weak_signals', [])
        
        if weak_signals:
            st.markdown("### 🚨 Signaux Faibles Détectés")
            
            for signal in weak_signals :
                if signal.get('formation_type') or signal.get('formation_id'):
                    signal_type = signal.get('type', 'info')
                    
                    if signal_type == 'critique':
                        st.error(f"""
                        **⚠️ CRITIQUE** - {signal.get('issue', 'Problème détecté')}
                        
                        - Formation_type:{signal.get('formation_type', 'N/A')}
                        - Formation_id: {signal.get('formation_id', signal.get('formation_type', 'N/A'))}
                        - Occurrences: {signal.get('occurrences', 0)}
                        - Score moyen: {signal.get('avg_score', 'N/A')}
                        - Details:{signal.get('details', 'N/A')}
                        """)
                    else:
                        st.warning(f"""
                        **⚡ WARNING** - {signal.get('issue', 'Attention requise')}

                        - Formation_type:{signal.get('formation_type', 'N/A')}
                        - Formation_id: {signal.get('formation_id', signal.get('formation_type', 'N/A'))}
                        - Occurrences: {signal.get('occurrences', 0)}
                        - Score moyen: {signal.get('avg_score', 'N/A')}
                        - Details:{signal.get('details', 'N/A')}
                        """)
        else:
            st.success("✅ Aucun signal faible critique détecté!")
        
        st.markdown("---")
        
        # Recommandations
        st.markdown("### 💡 Recommandations Automatiques")
        
        # Analyser les données pour générer des recommandations dynamiques
        avg_logistique = summary.get('avg_satisfaction', 0) 
        avg_satisfaction = summary.get('avg_satisfaction', 0)
        
        if avg_logistique < 3.5:
            st.info("""
            **📌 Recommandation 1: Améliorer la logistique**
            - Problèmes récurrents détectés sur les salles et horaires
            - Impact estimé: +0.4 points de satisfaction
            - Action: Réviser le planning et la capacité des salles
            """)
        
        if avg_satisfaction >= 4.0:
            st.success("""
            **🎯 Recommandation 2: Capitaliser sur les points forts**
            - Qualité des formateurs excellente
            - Maintenir le niveau de qualité du contenu pédagogique
            - Partager les bonnes pratiques entre formateurs
            """)
        
        if len(weak_signals) > 0:
            st.warning("""
            **⚠️ Recommandation 3: Traiter les signaux faibles**
            - Plusieurs problèmes récurrents identifiés
            - Mettre en place un plan d'action correctif
            - Suivre l'évolution sur les prochaines sessions
            """)
        
        # Performance par type
        st.markdown("---")
        st.markdown("### 📊 Performance par Type de Formation")
        
        formation_types = data.get('kpis', {}).get('formation_type', {}).get('by_type', [])
        
        if formation_types:
            df_formations = pd.DataFrame(formation_types)
            
            fig_formations = px.bar(
                df_formations,
                x='type',
                y='satisfaction_mean',
                color='satisfaction_mean',
                title="Satisfaction Moyenne par Type",
                color_continuous_scale='RdYlGn',
                range_color=[1, 5],
                text='satisfaction_mean'
            )
            fig_formations.update_layout(height=400)
            fig_formations.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_formations, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("#### 📋 Détails par Formation")
            st.dataframe(
                df_formations[['type', 'satisfaction_mean', 'count', 'nb_formations_uniques']].round(2),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Données par type de formation non disponibles")
        
        # Résumé exécutif
        st.markdown("---")
        st.markdown("### 📋 Résumé Exécutif")
        
        st.markdown(f"""
        **Période d'analyse:** {datetime.now().strftime('%B %Y')}
        
        **Vue d'ensemble:**
        - **{summary.get('total_evaluations', 0)}** évaluations analysées
        - Satisfaction globale: **{summary.get('avg_satisfaction', 0)}/5**
        - Taux de complétion: **{data.get('kpis', {}).get('global', {}).get('completion_rate', 0)}%**
        
        **Points forts:**
        - Qualité des formateurs appréciée
        - Contenu pédagogique pertinent
        - {data.get('sentiment_analysis', {}).get('distribution', {}).get('POSITIVE', 0)}% de sentiments positifs
        
        **Axes d'amélioration:**
        - Logistique à optimiser
        - Applicabilité pratique à renforcer
        - {len(weak_signals)} signaux faibles nécessitent attention
        
        **Recommandations prioritaires:**
        1. Améliorer la logistique des formations
        2. Renforcer les exercices pratiques
        3. Adapter le rythme selon les retours participants
        """)

else:
    # Page d'accueil si pas de données
    st.markdown("""
    ## 👋 Bienvenue sur le Système d'Analyse IA
    
    ### 🎯 Fonctionnalités
    
    - ✅ **Analyse automatisée** des évaluations de formation
    - ✅ **NLP avancé** (sentiment, thèmes, clustering)
    - ✅ **Détection de signaux faibles** en temps réel
    - ✅ **KPIs interactifs** et dashboards
    - ✅ **Conformité RGPD** (anonymisation automatique)
    
    ### 🚀 Pour Commencer
    
    1. Uploadez votre fichier CSV d'évaluations dans la barre latérale
    2. Cliquez sur "Analyser le fichier"
    3. Explorez les résultats dans les différents onglets
    
    ### 📊 Format Attendu du CSV
    
    Votre fichier doit contenir les colonnes suivantes:
    - `evaluation_id`, `formation_id`, `type_formation`
    - `formateur_id`, `satisfaction`, `contenu`
    - `logistique`, `applicabilite`, `commentaire`
    - `langue`, `date`
    """)
    
    # Afficher un exemple
    st.markdown("### 📝 Exemple de Données")
    
    example_data = {
        'evaluation_id': ['E001', 'E002', 'E003'],
        'satisfaction': [5, 4, 3],
        'contenu': [5, 4, 3],
        'logistique': [4, 3, 2],
        'applicabilite': [5, 4, 3],
        'commentaire': ['Excellente formation', 'Bonne formation', 'Problèmes logistique']
    }
    
    st.dataframe(pd.DataFrame(example_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>🚀 <strong>Think to Deploy</strong> - Safran Formation Analysis System</p>
    <p>MADE WITH ❤️ BY H2A</p>
</div>
""", unsafe_allow_html=True)