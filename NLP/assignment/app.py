import streamlit as st
import pickle
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict

# Load model data
@st.cache_resource
def load_model():
    """Load the trained n-gram model"""
    try:
        with open('ngram_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file 'ngram_model.pkl' not found. Please run the notebook first to generate the model.")
        return None

# Preprocessing function
def preprocess_text(text, remove_stopwords=False):
    """Preprocess text: lowercase, remove punctuation, tokenize"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 0]
    return tokens

# Prediction function
def predict_next_words(input_text, bigram_probs, trigram_probs, top_k=5, use_trigram=True):
    """Predict the next likely words given user input"""
    input_tokens = preprocess_text(input_text, remove_stopwords=False)
    
    if not input_tokens:
        return []
    
    suggestions = []
    
    if use_trigram and len(input_tokens) >= 2:
        context = (input_tokens[-2], input_tokens[-1])
        
        if context in trigram_probs:
            probs = trigram_probs[context]
            suggestions = [(word, data['probability']) for word, data in probs.items()]
            suggestions.sort(key=lambda x: x[1], reverse=True)
            suggestions = suggestions[:top_k]
        else:
            last_word = input_tokens[-1]
            if last_word in bigram_probs:
                probs = bigram_probs[last_word]
                suggestions = [(word, data['probability']) for word, data in probs.items()]
                suggestions.sort(key=lambda x: x[1], reverse=True)
                suggestions = suggestions[:top_k]
    else:
        last_word = input_tokens[-1]
        
        if last_word in bigram_probs:
            probs = bigram_probs[last_word]
            suggestions = [(word, data['probability']) for word, data in probs.items()]
            suggestions.sort(key=lambda x: x[1], reverse=True)
            suggestions = suggestions[:top_k]
    
    return suggestions

# Streamlit UI
st.set_page_config(
    page_title="Wikipedia Text Next Word Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .title {
        color: #1e3a8a;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        color: #475569;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .dataset-info {
        background-color: #dbeafe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e40af;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🌐 Wikipedia Text Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Wikipedia Content - Advanced N-gram Language Models</div>', unsafe_allow_html=True)

# Dataset Information Banner
st.markdown("""
    <div class="dataset-info">
    <strong>📊 Dataset:</strong> Wikipedia Articles | <strong>Coverage:</strong> Science, Technology, History, Culture & More | <strong>Model Type:</strong> Bigram & Trigram
    </div>
""", unsafe_allow_html=True)

# Load model
model_data = load_model()

if model_data is not None:
    bigram_probs = model_data['bigram_probs']
    trigram_probs = model_data['trigram_probs']
    vocabulary = model_data['vocabulary']
    
    # Display stats dashboard above sidebar
    st.markdown("---")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("📚 Vocabulary", f"{len(vocabulary):,}", delta="Total Words")
    with stats_col2:
        st.metric("🔗 Bigrams", f"{len(bigram_probs):,}", delta="Contexts")
    with stats_col3:
        st.metric("⛓️ Trigrams", f"{len(trigram_probs):,}", delta="Contexts")
    with stats_col4:
        total_contexts = len(bigram_probs) + len(trigram_probs)
        st.metric("🎯 Total", f"{total_contexts:,}", delta="Contexts")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration & Settings")
        
        st.subheader("Model Selection")
        use_trigram = st.toggle(
            "Use Trigram Model",
            value=True,
            help="Trigram: Uses last 2 words for better context (Recommended)\nBigram: Uses last 1 word for faster predictions"
        )
        
        st.subheader("Prediction Settings")
        top_k = st.slider(
            "Number of Suggestions",
            min_value=1,
            max_value=10,
            value=5,
            help="How many next word suggestions to display"
        )
        
        st.divider()
        st.subheader("📊 Model Statistics")
        
        # Create visualization for model statistics
        stats_data = {
            'Metric': ['Vocabulary', 'Bigram Contexts', 'Trigram Contexts'],
            'Count': [len(vocabulary), len(bigram_probs), len(trigram_probs)]
        }
        df_stats = pd.DataFrame(stats_data)
        
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.metric("Vocabulary Size", f"{len(vocabulary):,}")
            st.metric("Bigram Contexts", f"{len(bigram_probs):,}")
            st.metric("Trigram Contexts", f"{len(trigram_probs):,}")
        
        with col2:
            fig_stats = px.bar(
                df_stats,
                x='Metric',
                y='Count',
                title='Model Overview',
                color='Metric',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            fig_stats.update_layout(height=250, showlegend=False, xaxis_title='', yaxis_title='Count')
            st.plotly_chart(fig_stats, use_container_width=True, config={'displayModeBar': False})
        
        st.divider()
        st.info("💡 **Tip:** Trigram models provide better predictions with longer context! Works best with Wikipedia-style text.")
    
    # Main content
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🔤 Enter Your Text")
        user_input = st.text_input(
            "Type a phrase or sentence:",
            placeholder="e.g., 'wikipedia is' or 'the internet' or 'information technology'",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        st.subheader("🎯 Active Model")
        model_type = "Trigram" if use_trigram else "Bigram"
        if use_trigram:
            st.success(f"**{model_type}**\n(2-word context)")
        else:
            st.warning(f"**{model_type}**\n(1-word context)")
    
    # Prediction
    if user_input:
        suggestions = predict_next_words(user_input, bigram_probs, trigram_probs, top_k=top_k, use_trigram=use_trigram)
        
        if suggestions:
            st.markdown("---")
            st.success(f"✅ Found {len(suggestions)} prediction(s)")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Visual", "📋 Table", "📈 Analysis"])
            
            with tab1:
                st.subheader("Next Word Suggestions")
                
                # Create visualizations
                col_chart1, col_chart2 = st.columns([1.5, 1])
                
                with col_chart1:
                    # Bar chart for probabilities
                    df_chart = pd.DataFrame({
                        'Word': [word for word, _ in suggestions],
                        'Probability': [prob for _, prob in suggestions],
                        'Percentage': [f"{prob*100:.1f}%" for _, prob in suggestions]
                    })
                    
                    fig = px.bar(
                        df_chart,
                        x='Word',
                        y='Probability',
                        title='Next Word Probabilities',
                        labels={'Word': 'Next Word', 'Probability': 'Probability Score'},
                        color='Probability',
                        color_continuous_scale='viridis',
                        text='Percentage'
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(height=350, showlegend=False, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    # Pie chart for probability distribution
                    fig_pie = px.pie(
                        values=[prob for _, prob in suggestions],
                        names=[word for word, _ in suggestions],
                        title='Probability Distribution',
                        hole=0.4
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Detailed cards for each suggestion
                st.subheader("📊 Detailed Analysis")
                for i, (word, prob) in enumerate(suggestions, 1):
                    with st.container(border=True):
                        col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1])
                        
                        with col1:
                            st.metric("Rank", f"#{i}")
                        with col2:
                            st.write(f"**Word:** `{word}`")
                        with col3:
                            st.metric("Prob", f"{prob:.4f}")
                        with col4:
                            # Color-coded confidence badge
                            if prob > 0.3:
                                st.success(f"✅ {prob*100:.1f}%")
                            elif prob > 0.1:
                                st.warning(f"⚠️ {prob*100:.1f}%")
                            else:
                                st.info(f"ℹ️ {prob*100:.1f}%")
                        
                        # Progress bar
                        st.progress(prob, text=f"Confidence Level: {prob*100:.2f}%")
            
            with tab2:
                # Create enhanced dataframe for table view
                df_results = pd.DataFrame([
                    {
                        'Rank': i,
                        'Next Word': word,
                        'Probability': f"{prob:.4f}",
                        'Percentage': f"{prob*100:.2f}%",
                        'Confidence': 'High' if prob > 0.3 else ('Medium' if prob > 0.1 else 'Low')
                    }
                    for i, (word, prob) in enumerate(suggestions, 1)
                ])
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Download button for results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            
            with tab3:
                st.write("**📈 Advanced Prediction Analysis**")
                
                # Calculate statistics
                max_prob = max([p for _, p in suggestions])
                min_prob = min([p for _, p in suggestions])
                avg_prob = sum([p for _, p in suggestions]) / len(suggestions)
                
                # Statistics cards
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Highest Prob", f"{max_prob:.4f}", delta=f"+{(max_prob-avg_prob)*100:.1f}%")
                with metric_cols[1]:
                    st.metric("Average Prob", f"{avg_prob:.4f}")
                with metric_cols[2]:
                    st.metric("Lowest Prob", f"{min_prob:.4f}", delta=f"-{(avg_prob-min_prob)*100:.1f}%")
                with metric_cols[3]:
                    st.metric("Confidence", f"{max_prob*100:.1f}%")
                
                st.divider()
                
                # Probability distribution histogram
                st.write("**Probability Distribution**")
                df_hist = pd.DataFrame({
                    'Word': [word for word, _ in suggestions],
                    'Probability': [prob for _, prob in suggestions]
                })
                
                fig_hist = px.bar(
                    df_hist,
                    x='Word',
                    y='Probability',
                    title='Word Probability Comparison',
                    labels={'Word': 'Next Word', 'Probability': 'Probability Score'},
                    color='Probability',
                    color_continuous_scale='blues',
                    text='Probability'
                )
                fig_hist.update_traces(textposition='auto', texttemplate='%{text:.4f}')
                fig_hist.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("⚠️ No suggestions found for this input. Try a different word or phrase.")
    else:
        st.info("👉 Start typing above to get next word predictions based on the academic dataset!")
    
    # Examples section
    st.markdown("---")
    st.subheader("💡 Try These Examples")
    st.caption("👇 Click any example to test the prediction model")
    
    example_cols = st.columns(4)
    examples = [
        ("📖 wikipedia is", "wikipedia is"),
        ("🌐 the internet", "the internet"),
        ("💻 information technology", "information technology"),
        ("🕸️ the world wide web", "the world wide web")
    ]
    
    for col, (label, example) in zip(example_cols, examples):
        with col:
            if st.button(
                label,
                use_container_width=True,
                key=f"btn_{example}",
                help=f"Click to predict next word after '{example}'"
            ):
                st.session_state.user_input = example
                st.rerun()
    
    # Info section
    st.markdown("---")
    with st.expander("ℹ️ About This Application", expanded=False):
        tab_info1, tab_info2, tab_info3 = st.tabs(["How It Works", "Dataset Info", "Model Details"])
        
        with tab_info1:
            st.markdown("""
            ### N-gram Language Models
            
            This application uses statistical language models to predict the next word based on previous words:
            
            **🔹 Bigram Model**
            - Uses the **previous 1 word** to predict the next word
            - Formula: P(wᵢ | wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
            - Faster but less context-aware
            
            **🔹 Trigram Model** (Recommended)
            - Uses the **previous 2 words** to predict the next word
            - Formula: P(wᵢ | wᵢ₋₂, wᵢ₋₁) = count(wᵢ₋₂, wᵢ₋₁, wᵢ) / count(wᵢ₋₂, wᵢ₋₁)
            - Better context understanding and more accurate predictions
            """)
        
        with tab_info2:
            st.markdown("""
            ### Dataset Information
            
            **📊 Wikipedia Content**
            - **Type**: Wikipedia Articles & Encyclopedia Entries
            - **Primary Coverage**: 
              - Science & Technology
              - History & Culture
              - Nature & Biology
              - Internet & Computing
              - General Knowledge
            
            **📈 Corpus Statistics**
            - Comprehensive Wikipedia dataset with diverse topics
            - Neutral point of view (NPOV) writing style
            - Rich vocabulary covering multiple domains
            - Educational and informative content patterns
            
            **🎯 Use Cases**
            - Wikipedia article completion
            - General knowledge text prediction
            - Encyclopedia entry writing assistance
            - Educational content generation
            """)
        
        with tab_info3:
            st.markdown(f"""
            ### Model Configuration
            
            **📋 Current Model Statistics**
            
            | Metric | Count |
            |--------|-------|
            | Vocabulary Size | {len(vocabulary):,} |
            | Bigram Contexts | {len(bigram_probs):,} |
            | Trigram Contexts | {len(trigram_probs):,} |
            
            **⚙️ Prediction Settings**
            - Number of suggestions: Configurable (1-10)
            - Probability threshold: Dynamic based on context
            - Fallback mechanism: Bigram when trigram context not found
            
            **✨ Features**
            - Real-time predictions
            - Confidence scoring
            - Multiple result formats (visual, table, analysis)
            - Example-based learning
            """)

    # Add interactive exploration dashboard
    st.markdown("---")
    with st.expander("🔍 Model Explorer & Statistics", expanded=False):
        explorer_tabs = st.tabs(["Word Frequency", "Context Analysis", "Performance Metrics"])
        
        with explorer_tabs[0]:
            st.write("**Most Common Words in Vocabulary**")
            
            # Get word frequencies from bigram contexts
            word_freq = Counter()
            for word in vocabulary:
                word_freq[word] += 1
            
            # Get top words
            top_words = dict(word_freq.most_common(20))
            df_freq = pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency'])
            
            fig_freq = px.bar(
                df_freq,
                x='Word',
                y='Frequency',
                title='Top 20 Words in Vocabulary',
                color='Frequency',
                color_continuous_scale='sunsetdark'
            )
            fig_freq.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_freq, use_container_width=True)
        
        with explorer_tabs[1]:
            st.write("**Model Statistics Summary**")
            
            # Context statistics
            metrics_data = {
                'Metric': [
                    'Vocabulary Size',
                    'Bigram Contexts',
                    'Trigram Contexts',
                    'Total Contexts',
                    'Avg Words per Context (Bigram)',
                    'Avg Words per Context (Trigram)'
                ],
                'Value': [
                    len(vocabulary),
                    len(bigram_probs),
                    len(trigram_probs),
                    len(bigram_probs) + len(trigram_probs),
                    round(sum(len(v) for v in bigram_probs.values()) / len(bigram_probs), 2) if bigram_probs else 0,
                    round(sum(len(v) for v in trigram_probs.values()) / len(trigram_probs), 2) if trigram_probs else 0
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # Model size comparison
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bigram Model Size", f"{len(bigram_probs):,} contexts")
            with col2:
                st.metric("Trigram Model Size", f"{len(trigram_probs):,} contexts")
        
        with explorer_tabs[2]:
            st.write("**Performance Overview**")
            
            perf_data = {
                'Model': ['Bigram', 'Trigram', 'Combined'],
                'Contexts': [len(bigram_probs), len(trigram_probs), len(bigram_probs) + len(trigram_probs)],
                'Type': ['Simple', 'Advanced', 'Hybrid']
            }
            df_perf = pd.DataFrame(perf_data)
            
            fig_perf = px.bar(
                df_perf,
                x='Model',
                y='Contexts',
                color='Type',
                title='Model Comparison',
                color_discrete_map={'Simple': '#FF6B6B', 'Advanced': '#4ECDC4', 'Hybrid': '#45B7D1'},
                text='Contexts'
            )
            fig_perf.update_traces(textposition='outside')
            fig_perf.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_perf, use_container_width=True)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔧 Built with Streamlit")
    with col2:
        st.caption("🌐 Dataset: Wikipedia")
    with col3:
        st.caption("🤖 Powered by N-gram Models")

else:
    st.error("❌ Failed to load the model. Please ensure 'ngram_model.pkl' exists in the same directory as this script.")
