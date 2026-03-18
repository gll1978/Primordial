# PRIMORDIAL EXPLORER DASHBOARD - Project Specification

**Date:** 2026-01-31  
**Author:** Gabry  
**Project Type:** Interactive Data Analytics Dashboard  
**Tech Stack:** Python + Streamlit + PostgreSQL + Plotly  
**Timeline:** 3 weeks  
**Deployment:** Streamlit Cloud (free tier)

---

## 🎯 PROJECT GOALS

### Primary Objective
Create an interactive web dashboard to explore and analyze data from 72 PRIMORDIAL V2 simulation runs containing 1.96M organisms and 110M learning events.

### Success Criteria
- ✅ Dashboard connects to existing PostgreSQL database
- ✅ Users can explore individual runs interactively
- ✅ Auto-generates insights and patterns
- ✅ Allows side-by-side run comparisons
- ✅ Exports publication-ready figures
- ✅ Deployed and publicly accessible
- ✅ Professional, clean UI/UX

### Secondary Benefits
- Solves "data overwhelming" problem with automated analysis
- Portfolio piece demonstrating full-stack capabilities
- Foundation for future work (papers, blog posts, talks)
- Fun and satisfying to build and use

---

## 📊 DATA SOURCE

### Database Connection
```yaml
Type: PostgreSQL
Host: localhost (development) / Cloud (production)
Database: primordial_v2
Tables:
  - organisms (1.96M rows)
  - organism_snapshots (110M rows)
  - learning_events (110M rows)
  - food_events
  - environment_state
  - runs (72 runs)
```

### Key Data Entities

**Organisms Table:**
```sql
organism_id, lineage_id, run_id, generation, brain_layers, 
num_neurons, num_connections, predator, aquatic, total_kills,
total_food_eaten, lifespan, birth_step, death_step, death_cause
```

**Learning Events Table:**
```sql
organism_id, step, event_type, reward, learning_magnitude,
outcome, brain_layers
```

**Organism Snapshots Table:**
```sql
organism_id, step, x, y, energy, brain_layers, is_day
```

### Derived Metrics
- Brain evolution timeline (avg layers over time)
- Species distribution (herbivore/predator/aquatic %)
- Lineage diversity and dominance
- Learning efficiency by brain complexity
- Survival rates and death causes
- Spatial distribution patterns
- Temporal patterns (day/night, seasonal)

---

## 🏗️ ARCHITECTURE

### Tech Stack

**Backend:**
- Python 3.11+
- SQLAlchemy (ORM for PostgreSQL)
- Pandas (data manipulation)
- NumPy (numerical operations)

**Frontend:**
- Streamlit (web framework)
- Plotly (interactive charts)
- Plotly Express (quick plots)
- Streamlit-Plotly-Events (interactivity)

**Deployment:**
- Streamlit Cloud (hosting)
- GitHub (code repository)
- PostgreSQL (cloud-hosted or tunneled)

### Project Structure
```
primordial-explorer/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── src/
│   ├── __init__.py
│   ├── database.py          # PostgreSQL connection & queries
│   ├── analysis.py          # Auto-analysis algorithms
│   ├── visualizations.py    # Plotly chart generators
│   └── exporters.py         # PDF/CSV/Figure exporters
├── pages/
│   ├── 1_Run_Explorer.py    # Individual run analysis
│   ├── 2_Comparison.py      # Multi-run comparison
│   ├── 3_Global_Stats.py    # Aggregate statistics
│   └── 4_Export.py          # Export tools
├── assets/
│   └── logo.png             # PRIMORDIAL logo
├── tests/
│   └── test_analysis.py     # Unit tests
└── README.md                # Documentation
```

---

## 📱 FEATURE SPECIFICATIONS

### 🏠 Home Page (app.py)

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  🧬 PRIMORDIAL V2 - Evolution Explorer                  │
│  Interactive Analysis Dashboard                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📊 Global Statistics                                   │
│  ├─ Total Runs: 72                                      │
│  ├─ Total Organisms: 1,956,043                          │
│  ├─ Learning Events: 110,661,305                        │
│  └─ Max Generation: 342                                 │
│                                                         │
│  🎲 Outcome Distribution                                │
│  ┌──────────────────────────────────────┐              │
│  │ Herbivore ████████████ 50%           │              │
│  │ Aquatic   ████████ 33%               │              │
│  │ Predator  ████ 17%                   │              │
│  └──────────────────────────────────────┘              │
│                                                         │
│  🔥 Recent Discoveries                                  │
│  • Lineage #269 dominance (81 generations)             │
│  • Cognitive compensation (Winter → +67% brain)        │
│  • Longevity-intelligence trade-off (4.8×)             │
│                                                         │
│  👉 Select a page from the sidebar to begin exploring  │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
import streamlit as st
import src.database as db

st.set_page_config(
    page_title="PRIMORDIAL Explorer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 PRIMORDIAL V2 - Evolution Explorer")
st.caption("Interactive Analysis Dashboard")

# Global stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Runs", db.get_total_runs())
with col2:
    st.metric("Total Organisms", f"{db.get_total_organisms():,}")
with col3:
    st.metric("Learning Events", f"{db.get_total_learning():,}")
with col4:
    st.metric("Max Generation", db.get_max_generation())

# Outcome distribution
st.subheader("🎲 Outcome Distribution")
outcomes = db.get_outcome_distribution()
fig = px.bar(outcomes, x='outcome', y='count', color='outcome')
st.plotly_chart(fig, use_container_width=True)

# Recent discoveries
st.subheader("🔥 Key Discoveries")
st.success("✓ Lineage #269 dominance (81 generations, brain 24.12)")
st.info("✓ Cognitive compensation (Winter → +67% brain size)")
st.warning("✓ Longevity-intelligence trade-off (simple brains live 4.8× longer)")
```

---

### 🔍 Page 1: Run Explorer

**Purpose:** Deep-dive into a single simulation run

**Features:**
1. **Run Selector**
   - Dropdown with all 72 runs
   - Shows: Run ID, dominant species, max generation, steps
   - Search/filter by characteristics

2. **Auto-Analysis Panel**
   - Automatically detects patterns
   - Generates insights in natural language
   - Highlights anomalies

3. **Evolution Timeline**
   - Brain complexity over time
   - Population dynamics
   - Species composition

4. **Lineage Explorer**
   - Top lineages by generation
   - Lineage #269 special highlight
   - Family tree visualization

5. **Ecology Panel**
   - Predator/prey ratio
   - Death causes breakdown
   - Spatial distribution heatmap

**Implementation Outline:**
```python
# pages/1_Run_Explorer.py

import streamlit as st
import src.database as db
import src.analysis as analysis
import src.visualizations as viz

st.title("🔍 Run Explorer")

# Run selector
runs = db.get_all_runs()
run_id = st.selectbox(
    "Select Run",
    options=runs['run_id'],
    format_func=lambda x: f"Run {x} - {db.get_run_summary(x)}"
)

# Auto-analysis
st.header("🤖 Auto-Generated Insights")
insights = analysis.auto_analyze_run(run_id)
for insight in insights:
    st.success(f"✓ {insight}")

# Brain evolution
st.header("🧠 Brain Evolution")
brain_data = db.get_brain_evolution(run_id)
fig = viz.plot_brain_evolution(brain_data)
st.plotly_chart(fig, use_container_width=True)

# Lineages
st.header("🌳 Lineage Analysis")
lineage_data = db.get_lineages(run_id)
st.dataframe(lineage_data)

# Ecology
st.header("🌍 Ecological Dynamics")
col1, col2 = st.columns(2)
with col1:
    death_causes = db.get_death_causes(run_id)
    fig = viz.plot_death_causes(death_causes)
    st.plotly_chart(fig)
with col2:
    species_dist = db.get_species_distribution(run_id)
    fig = viz.plot_species_pie(species_dist)
    st.plotly_chart(fig)
```

---

### 📊 Page 2: Run Comparison

**Purpose:** Side-by-side analysis of multiple runs

**Features:**
1. **Multi-Select Runs** (2-6 runs)
2. **Comparison Metrics Table**
   - Dominant species
   - Max generation
   - Brain evolution (start → end)
   - Lineage diversity
   - Death rates
3. **Overlay Charts**
   - Brain evolution (all runs on same plot)
   - Population timeline comparison
4. **Divergence Analysis**
   - When did runs diverge?
   - What caused different outcomes?
   - Auto-generated hypothesis

**Implementation Outline:**
```python
# pages/2_Comparison.py

st.title("📊 Run Comparison")

# Multi-select
selected_runs = st.multiselect(
    "Select runs to compare (2-6)",
    options=db.get_all_runs()['run_id'],
    max_selections=6
)

if len(selected_runs) >= 2:
    # Comparison table
    st.header("📋 Comparison Table")
    comparison = db.get_comparison_data(selected_runs)
    st.dataframe(comparison)
    
    # Overlay chart
    st.header("🧠 Brain Evolution Overlay")
    brain_data = db.get_brain_evolution_multi(selected_runs)
    fig = viz.plot_brain_overlay(brain_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Divergence analysis
    st.header("🔀 Divergence Analysis")
    divergence = analysis.analyze_divergence(selected_runs)
    st.write(divergence['narrative'])
else:
    st.info("Select at least 2 runs to compare")
```

---

### 📈 Page 3: Global Statistics

**Purpose:** Aggregate analysis across all 72 runs

**Features:**
1. **Outcome Distribution**
   - Herbivore: 50% (36 runs)
   - Aquatic: 33% (24 runs)
   - Predator: 17% (12 runs)
   - Stochastic Assembly visualization

2. **Discovery #10 Validation**
   - Same parameters → Different outcomes
   - Early events predict outcome (0-1000 steps)
   - Statistical analysis

3. **Meta-Patterns**
   - What correlates with success?
   - Brain size vs survival
   - Predation vs dominance
   - Lineage diversity vs stability

4. **Aggregate Metrics**
   - Total organisms born: 1.96M
   - Total learning events: 110M
   - Average brain evolution: 7.96 → 28.36
   - Species equilibrium: 92% herbivore, 8% predator

**Implementation Outline:**
```python
# pages/3_Global_Stats.py

st.title("📈 Global Statistics")
st.caption("Aggregate analysis across all 72 runs")

# Outcome distribution
st.header("🎲 Stochastic Assembly (Discovery #10)")
outcomes = db.get_outcome_distribution()
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Herbivore Dominance", "50%", "36 runs")
    st.metric("Aquatic Dominance", "33%", "24 runs")
    st.metric("Predator Dominance", "17%", "12 runs")
with col2:
    fig = viz.plot_outcome_pie(outcomes)
    st.plotly_chart(fig)

st.success("✓ Same parameters → Divergent outcomes (validated)")

# Meta-patterns
st.header("🔬 Meta-Patterns Across Runs")
patterns = analysis.find_meta_patterns()
for pattern in patterns:
    st.info(f"• {pattern}")

# Aggregate metrics
st.header("📊 Aggregate Metrics")
metrics = db.get_aggregate_metrics()
st.dataframe(metrics)
```

---

### 💾 Page 4: Export Tools

**Purpose:** Generate publication-ready outputs

**Features:**
1. **PDF Report Generator**
   - Select run(s)
   - Choose template (paper/presentation/blog)
   - Auto-generate comprehensive report

2. **Figure Exporter**
   - Publication-quality PNG/SVG
   - Pre-configured for Nature/Science specs
   - Batch export for all key figures

3. **Data Exporter**
   - CSV exports for specific queries
   - Filtered datasets
   - Aggregate statistics

4. **Paper Figure Pack**
   - Figure 1: Brain Evolution
   - Figure 2: Learning Efficiency
   - Figure 3: Seasonal Compensation
   - Figure 4: Lineage #269
   - Figure 5: Longevity Trade-off
   - Figure 6: Stochastic Assembly

**Implementation Outline:**
```python
# pages/4_Export.py

st.title("💾 Export Tools")

# PDF Report
st.header("📄 PDF Report Generator")
run_ids = st.multiselect("Select runs", db.get_all_runs()['run_id'])
template = st.radio("Template", ["Paper", "Presentation", "Blog Post"])
if st.button("Generate Report"):
    pdf = exporters.generate_pdf_report(run_ids, template)
    st.download_button("Download PDF", pdf, "primordial_report.pdf")

# Figure exporter
st.header("📊 Publication Figures")
figure_type = st.selectbox("Select Figure", [
    "Brain Evolution Timeline",
    "Learning Efficiency",
    "Seasonal Compensation",
    "Lineage #269 Analysis",
    "Longevity Trade-off",
    "Stochastic Assembly"
])
format = st.radio("Format", ["PNG (300 DPI)", "SVG (Vector)"])
if st.button("Generate Figure"):
    fig = exporters.generate_figure(figure_type, format)
    st.pyplot(fig)
    st.download_button("Download", fig, f"{figure_type}.{format.lower()}")

# Data export
st.header("📊 Data Export")
query_type = st.selectbox("Select Dataset", [
    "Brain Evolution (all runs)",
    "Lineage Data",
    "Learning Events Summary",
    "Death Causes",
    "Custom Query"
])
if st.button("Export CSV"):
    csv = exporters.export_csv(query_type)
    st.download_button("Download CSV", csv, "data.csv")
```

---

## 🔧 TECHNICAL IMPLEMENTATION

### Database Module (src/database.py)

```python
"""
Database connection and query functions for PRIMORDIAL Explorer
"""

import os
from sqlalchemy import create_engine, text
import pandas as pd
from typing import List, Dict, Optional

class PrimordialDB:
    """PostgreSQL database connector for PRIMORDIAL V2"""
    
    def __init__(self):
        # Connection string from environment or config
        db_url = os.getenv('DATABASE_URL', 'postgresql://localhost/primordial_v2')
        self.engine = create_engine(db_url)
    
    def get_total_runs(self) -> int:
        """Get total number of runs in database"""
        query = "SELECT COUNT(DISTINCT run_id) FROM organisms"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.scalar()
    
    def get_total_organisms(self) -> int:
        """Get total organisms ever born"""
        query = "SELECT COUNT(*) FROM organisms"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.scalar()
    
    def get_total_learning(self) -> int:
        """Get total learning events"""
        query = "SELECT COUNT(*) FROM learning_events"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.scalar()
    
    def get_brain_evolution(self, run_id: str) -> pd.DataFrame:
        """Get brain evolution timeline for a run"""
        query = """
        SELECT 
            (step / 10000) as time_k,
            AVG(brain_layers) as avg_layers,
            STDDEV(brain_layers) as std_layers,
            MIN(brain_layers) as min_layers,
            MAX(brain_layers) as max_layers,
            COUNT(*) as population
        FROM organism_snapshots
        WHERE step % 10000 = 0
        GROUP BY time_k
        ORDER BY time_k
        """
        return pd.read_sql(query, self.engine)
    
    def get_lineages(self, run_id: str) -> pd.DataFrame:
        """Get top lineages for a run"""
        query = """
        SELECT 
            lineage_id,
            COUNT(*) as total_organisms,
            MAX(generation) as max_generation,
            AVG(brain_layers) as avg_brain,
            AVG(num_neurons) as avg_neurons,
            SUM(CASE WHEN predator THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_predators
        FROM organisms
        GROUP BY lineage_id
        ORDER BY max_generation DESC
        LIMIT 20
        """
        return pd.read_sql(query, self.engine)
    
    def get_outcome_distribution(self) -> pd.DataFrame:
        """Get distribution of run outcomes (herbivore/aquatic/predator dominance)"""
        # This requires run metadata - implement based on your schema
        pass
    
    # Add more query methods as needed...

# Singleton instance
db = PrimordialDB()
```

### Analysis Module (src/analysis.py)

```python
"""
Auto-analysis algorithms for pattern detection
"""

from typing import List, Dict
import pandas as pd
import numpy as np

def auto_analyze_run(run_id: str) -> List[str]:
    """
    Automatically analyze a run and generate insights
    
    Returns:
        List of insight strings
    """
    insights = []
    
    # Get brain evolution
    brain_data = db.get_brain_evolution(run_id)
    start_brain = brain_data['avg_layers'].iloc[0]
    end_brain = brain_data['avg_layers'].iloc[-1]
    growth = end_brain - start_brain
    
    if growth > 20:
        insights.append(f"Brain evolved from {start_brain:.1f} → {end_brain:.1f} layers (+{growth:.1f})")
    
    # Check for lineage dominance
    lineages = db.get_lineages(run_id)
    dominant_lineage = lineages.iloc[0]
    
    if dominant_lineage['max_generation'] > 50:
        insights.append(
            f"Lineage #{dominant_lineage['lineage_id']} dominated with "
            f"{dominant_lineage['max_generation']} generations"
        )
    
    # Check for non-predatory success
    if dominant_lineage['pct_predators'] < 1:
        insights.append(
            f"Dominant lineage is NON-predatory (challenges predation-intelligence hypothesis)"
        )
    
    # Check death causes
    death_causes = db.get_death_causes(run_id)
    starvation_pct = death_causes[death_causes['cause']=='Starvation']['percentage'].values[0]
    
    if starvation_pct > 50:
        insights.append(
            f"Starvation is primary death cause ({starvation_pct:.1f}%) - "
            f"competition > predation"
        )
    
    # Add more pattern detection...
    
    return insights

def analyze_divergence(run_ids: List[str]) -> Dict:
    """
    Analyze when and why runs diverged
    
    Returns:
        Dictionary with divergence analysis
    """
    # Implementation: compare early events (0-1000 steps)
    # Find critical decision points
    # Generate narrative explanation
    pass

def find_meta_patterns() -> List[str]:
    """
    Find patterns across all runs
    
    Returns:
        List of meta-pattern insights
    """
    patterns = []
    
    # Pattern: Brain size vs success
    # Pattern: Predation rate vs outcome
    # Pattern: Early events predict outcome
    # etc.
    
    return patterns
```

### Visualization Module (src/visualizations.py)

```python
"""
Plotly chart generators
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_brain_evolution(data: pd.DataFrame) -> go.Figure:
    """
    Create brain evolution timeline chart
    
    Args:
        data: DataFrame with columns [time_k, avg_layers, std_layers, min_layers, max_layers]
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=data['time_k'],
        y=data['avg_layers'],
        mode='lines+markers',
        name='Average',
        line=dict(color='blue', width=3)
    ))
    
    # Std dev band
    fig.add_trace(go.Scatter(
        x=data['time_k'],
        y=data['avg_layers'] + data['std_layers'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=data['time_k'],
        y=data['avg_layers'] - data['std_layers'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        name='Std Dev'
    ))
    
    # Min/max
    fig.add_trace(go.Scatter(
        x=data['time_k'],
        y=data['max_layers'],
        mode='lines',
        name='Maximum',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Brain Evolution Timeline',
        xaxis_title='Time (k steps)',
        yaxis_title='Brain Layers',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_death_causes(data: pd.DataFrame) -> go.Figure:
    """Create pie chart of death causes"""
    fig = px.pie(
        data,
        values='count',
        names='cause',
        title='Death Causes Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig

def plot_species_pie(data: pd.DataFrame) -> go.Figure:
    """Create pie chart of species distribution"""
    fig = px.pie(
        data,
        values='count',
        names='species',
        title='Species Distribution',
        color_discrete_map={
            'herbivore': 'green',
            'predator': 'red',
            'aquatic': 'blue'
        }
    )
    return fig

# Add more visualization functions...
```

### Export Module (src/exporters.py)

```python
"""
Export tools for PDF reports, figures, and data
"""

from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report(run_ids: List[str], template: str) -> bytes:
    """
    Generate PDF report for selected runs
    
    Args:
        run_ids: List of run IDs to include
        template: 'Paper', 'Presentation', or 'Blog Post'
    
    Returns:
        PDF bytes
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Title page
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 750, "PRIMORDIAL V2 - Analysis Report")
    
    # Add content based on template
    # ...
    
    c.save()
    buffer.seek(0)
    return buffer.read()

def generate_figure(figure_type: str, format: str) -> bytes:
    """
    Generate publication-quality figure
    
    Args:
        figure_type: Type of figure to generate
        format: 'PNG' or 'SVG'
    
    Returns:
        Figure bytes
    """
    # Create matplotlib figure with publication specs
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Generate specific figure based on type
    # ...
    
    buffer = BytesIO()
    plt.savefig(buffer, format=format.lower(), bbox_inches='tight')
    buffer.seek(0)
    return buffer.read()

def export_csv(query_type: str) -> bytes:
    """
    Export data as CSV
    
    Args:
        query_type: Type of data to export
    
    Returns:
        CSV bytes
    """
    # Execute appropriate query
    # ...
    
    buffer = BytesIO()
    data.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.read()
```

---

## 📦 DEPENDENCIES

### requirements.txt
```txt
# Core
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Export
reportlab>=4.0.0
Pillow>=10.0.0

# Optional
streamlit-plotly-events>=0.0.6
streamlit-aggrid>=0.3.4
```

---

## 🚀 DEPLOYMENT

### Streamlit Cloud Setup

1. **Prepare Repository**
```bash
git init
git add .
git commit -m "Initial commit: PRIMORDIAL Explorer"
git remote add origin <github-url>
git push -u origin main
```

2. **Streamlit Cloud Configuration**
   - Go to https://share.streamlit.io
   - Connect GitHub repository
   - Select main file: `app.py`
   - Add secrets for database connection

3. **Database Connection**
   - Option A: Cloud PostgreSQL (Supabase, Neon, etc.)
   - Option B: SSH tunnel to local database
   - Option C: Replicate database to cloud

4. **Secrets Configuration**
```toml
# .streamlit/secrets.toml (not committed to git)
[database]
url = "postgresql://user:password@host:port/primordial_v2"
```

---

## 📋 IMPLEMENTATION PHASES

### Week 1: Core Features (MVP)

**Day 1-2: Setup**
- [ ] Create project structure
- [ ] Setup database connection
- [ ] Test basic queries
- [ ] Create Home page

**Day 3-4: Run Explorer**
- [ ] Implement run selector
- [ ] Brain evolution chart
- [ ] Basic auto-analysis
- [ ] Lineage table

**Day 5-7: Visualizations**
- [ ] Death causes chart
- [ ] Species distribution
- [ ] Population timeline
- [ ] Polish UI

**Deliverable:** Working dashboard with single-run exploration

---

### Week 2: Advanced Features

**Day 1-2: Comparison**
- [ ] Multi-run selector
- [ ] Comparison table
- [ ] Overlay charts
- [ ] Divergence analysis

**Day 3-4: Global Stats**
- [ ] Outcome distribution
- [ ] Meta-patterns
- [ ] Aggregate metrics
- [ ] Discovery validation

**Day 5-7: Auto-Analysis**
- [ ] Pattern detection algorithms
- [ ] Insight generation
- [ ] Anomaly detection
- [ ] Narrative generation

**Deliverable:** Full-featured dashboard with comparison and insights

---

### Week 3: Polish & Deploy

**Day 1-2: Export Tools**
- [ ] PDF report generator
- [ ] Figure exporter (PNG/SVG)
- [ ] CSV data export
- [ ] Paper figure pack

**Day 3-4: UI/UX Polish**
- [ ] Consistent styling
- [ ] Loading indicators
- [ ] Error handling
- [ ] Responsive design
- [ ] Help tooltips

**Day 5: Documentation**
- [ ] README with screenshots
- [ ] User guide
- [ ] API documentation
- [ ] Example queries

**Day 6-7: Deployment**
- [ ] Deploy to Streamlit Cloud
- [ ] Test all features live
- [ ] Fix deployment issues
- [ ] Share public URL

**Deliverable:** Production-ready dashboard deployed and accessible

---

## ✅ TESTING CHECKLIST

### Functionality Tests
- [ ] Database connection works
- [ ] All queries return data
- [ ] Charts render correctly
- [ ] Export functions work
- [ ] No errors in console

### UI/UX Tests
- [ ] Responsive on mobile
- [ ] Fast loading (<3s)
- [ ] Intuitive navigation
- [ ] Clear error messages
- [ ] Professional appearance

### Data Quality Tests
- [ ] Correct calculations
- [ ] No missing data
- [ ] Consistent formatting
- [ ] Accurate insights

---

## 🎯 SUCCESS METRICS

### Technical
- [ ] Dashboard loads in <3 seconds
- [ ] All 72 runs accessible
- [ ] Charts interactive and responsive
- [ ] Export functions working
- [ ] Deployed and publicly accessible

### User Experience
- [ ] Solves "data overwhelming" problem
- [ ] Insights auto-generated correctly
- [ ] Easy to explore and compare runs
- [ ] Publication-ready exports
- [ ] Professional appearance

### Business Value
- [ ] Portfolio piece (shareable URL)
- [ ] Foundation for papers/blogs
- [ ] Demonstrates full-stack skills
- [ ] Enables future work

---

## 📚 RESOURCES

### Documentation
- Streamlit: https://docs.streamlit.io
- Plotly: https://plotly.com/python/
- SQLAlchemy: https://docs.sqlalchemy.org
- Pandas: https://pandas.pydata.org/docs/

### Examples
- Streamlit Gallery: https://streamlit.io/gallery
- Plotly Examples: https://plotly.com/python/
- Dashboard Templates: https://github.com/streamlit/example-app-template

### Database Schema Reference
- See PRIMORDIAL V2 documentation
- Tables: organisms, organism_snapshots, learning_events
- Key relationships: organism_id, lineage_id, run_id

---

## 🔮 FUTURE ENHANCEMENTS (Post-MVP)

### Phase 4 (Optional)
- Real-time simulation monitoring
- Custom query builder
- Machine learning predictions
- 3D spatial visualizations
- Video generation of evolution
- API for programmatic access
- User accounts and saved views
- Collaborative features (comments, annotations)

---

## 📞 SUPPORT & MAINTENANCE

### Known Issues
- Database connection may timeout (add retry logic)
- Large queries may be slow (add caching)
- Export PDFs need styling improvements

### Maintenance Plan
- Weekly: Check for errors in logs
- Monthly: Update dependencies
- Quarterly: Add new features based on usage

---

## 🎉 PROJECT COMPLETION

When this project is complete, you will have:

✅ **A working interactive dashboard** that solves the "data overwhelming" problem  
✅ **A portfolio piece** demonstrating full-stack skills (Rust + PostgreSQL + Python + Web)  
✅ **A foundation** for papers, blog posts, and talks  
✅ **Publication-ready exports** for Nature/Science submissions  
✅ **Public URL** to share with others  
✅ **Satisfaction** of building something concrete and useful!

---

**Ready to start building? Let's go! 🚀**

*Document Version: 1.0*  
*Last Updated: 2026-01-31*
