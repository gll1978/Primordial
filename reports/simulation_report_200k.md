# PRIMORDIAL V2 - Report Simulazione 200k Step

**Data**: 2026-01-31
**Durata**: ~20 ore
**Step totali**: 200,000

---

## Riepilogo Esecutivo

| Metrica | Valore |
|---------|--------|
| Popolazione finale | 6,146 |
| Generazione massima | 342 |
| Brain medio finale | 30.0 (MAX) |
| Lineages sopravvissuti | 3 |
| Specie dominante | **Herbivore** (96%) |

---

## 1. Statistiche Generali

| Metrica | Valore |
|---------|--------|
| Organismi totali (storico) | 1,956,043 |
| Vivi | 132,939 |
| Morti | 1,823,104 |
| Lineages totali | 5,080 |
| Run nel database | 72 |

---

## 2. Top 15 Lineages per Generazione

| Lineage | Organismi | Max Gen | Avg Brain | Avg Neurons | % Predatori |
|---------|-----------|---------|-----------|-------------|-------------|
| **#269** | 53,731 | **81** | **24.12** | 95.7 | 0.0% |
| #334 | 1,166 | 74 | 13.34 | 56.1 | 0.2% |
| #372 | 625 | 62 | 13.07 | 53.5 | 0.0% |
| #2194 | 64,643 | 61 | 17.47 | 66.7 | 0.6% |
| #4813 | 53,248 | 58 | 17.86 | 73.5 | 0.3% |
| #1043 | 86,193 | 58 | 17.39 | 65.9 | 0.2% |
| #1501 | 30,946 | 57 | 16.03 | 65.0 | 0.1% |
| #866 | 51,994 | 57 | 15.78 | 61.8 | 0.3% |
| #396 | 22,979 | 57 | 18.97 | 76.9 | 0.0% |
| #1963 | 104,885 | 57 | 16.91 | 68.2 | 0.0% |

**Insight**: Lineage #269 domina con 81 generazioni e cervelli 2x più complessi della media.

---

## 3. Lineage #269 vs Altri

| Metrica | Lineage #269 | Altri | Differenza |
|---------|--------------|-------|------------|
| Avg Brain Layers | **24.12** | 11.36 | **+112%** |
| Avg Neurons | **95.7** | 44.8 | **+114%** |
| Avg Connections | 983.0 | 711.6 | +38% |
| Avg Offspring | 0.989 | 0.926 | +7% |
| Avg Kills | 0.012 | 0.354 | **-97%** |
| Avg Food Eaten | 561.0 | 1704.5 | -67% |
| Avg Lifespan | 10,574 | 14,345 | -26% |

**Strategia vincente**: Cervelli complessi, NO predazione, efficienza alimentare.

---

## 4. Predatori vs Non-Predatori

| Metrica | Predatori | Non-Predatori |
|---------|-----------|---------------|
| Totale | 165,352 (8.5%) | 1,790,691 (91.5%) |
| Avg Brain | 13.96 | 11.50 |
| Avg Neurons | 53.5 | 45.6 |
| Avg Offspring | 0.966 | 0.924 |
| **Avg Lifespan** | **6,676** | **14,940** |
| Avg Kills | 3.81 | 0.02 |
| Peak Energy | 133.0 | 133.0 |

**Insight**: Predatori hanno cervelli +21% più complessi ma vivono **-55% meno**.

---

## 5. Evoluzione nel Tempo

| Step (k) | Popolazione | % Predatori | Avg Brain | Avg Energy |
|----------|-------------|-------------|-----------|------------|
| 0 | 13,004,440 | 10.24% | **7.96** | 245.8 |
| 50 | 4,690,034 | 8.04% | 14.16 | 350.8 |
| 100 | 3,657,600 | 7.04% | 19.44 | 361.4 |
| 150 | 3,657,600 | 6.93% | 24.51 | 349.1 |
| 200 | 3,657,600 | 7.76% | **28.36** | 362.9 |

**Trend**: Brain cresciuto da **8 → 28 layers** (+250%). Predatori stabili al 7-8%.

---

## 6. Complessita Cerebrale vs Successo

| Brain Layers | Totale | Avg Offspring | Avg Lifespan | % Predatori |
|--------------|--------|---------------|--------------|-------------|
| 1-9 | 913,226 | 0.945 | **22,532** | 6.31% |
| 10-14 | 481,147 | 0.877 | 7,956 | 10.15% |
| 15-19 | 231,387 | 0.942 | 6,901 | 9.64% |
| 20-24 | 145,397 | **0.960** | 5,822 | 10.36% |
| 25-29 | 103,872 | 0.958 | 4,708 | 6.96% |
| 30+ | 81,014 | 0.883 | 6,406 | **17.70%** |

**Trade-off**: Cervelli semplici (1-9) vivono 4x piu a lungo. Cervelli 30+ hanno 2.5x piu predatori.

---

## 7. Cause di Morte

| Causa | Totale | % | Avg Lifespan |
|-------|--------|---|--------------|
| **Starvation** | 1,137,656 | **62.4%** | 258 |
| Old Age | 618,899 | 34.0% | 5,000 |
| Predation | 64,124 | 3.5% | 609 |
| Unknown | 2,425 | 0.1% | 39 |

**Insight**: Fame e la causa principale di morte (62%). Solo 3.5% muore per predazione.

---

## 8. Attivita Giorno vs Notte

| Periodo | Snapshots | Avg Energy | Avg Brain |
|---------|-----------|------------|-----------|
| Day | 55,655,488 | 348.4 | 16.40 |
| Night | 55,008,846 | 340.4 | 16.65 |

**Differenza**: Energia -2.3% di notte. Attivita simile.

---

## 9. Diversita Genetica nel Tempo

| Step (k) | Lineages Attivi | Nascite | Brain StdDev |
|----------|-----------------|---------|--------------|
| 0 | 5,080 | 883,461 | 4.16 |
| 50 | 58 | 84,167 | 7.22 |
| 100 | 21 | 27,139 | 2.93 |
| 150 | 21 | 26,473 | 3.16 |
| 200 | 21 | 25,424 | 2.11 |

**Trend**: Lineages calati da 5080 → 21 (-99.6%). Forte selezione naturale.

---

## 10. Top Organismi

### Per Offspring (Top 5)

| ID | Lineage | Gen | Offspring | Brain | Lifespan |
|----|---------|-----|-----------|-------|----------|
| 21252 | #269 | 69 | **4,538** | 30 | 5,000 |
| 21256 | #269 | 70 | 3,027 | 30 | 5,000 |
| 20774 | #269 | 71 | 2,881 | 30 | 5,000 |
| 20769 | #269 | 70 | 2,871 | 30 | 5,000 |
| 20776 | #269 | 72 | 2,583 | 30 | 5,000 |

### Per Kills (Top 5)

| ID | Lineage | Gen | Kills | Brain | Lifespan |
|----|---------|-----|-------|-------|----------|
| 8034 | #105 | 13 | **287** | 5 | 1,046 |
| 8028 | #105 | 13 | 287 | 8 | 1,046 |
| 8030 | #105 | 13 | 287 | 6 | 1,048 |
| 10068 | #105 | 14 | 286 | 8 | 1,171 |
| 10063 | #105 | 14 | 286 | 7 | 1,171 |

---

## 11. Distribuzione Run Outcomes

| Specie Dominante | Frequenza | % | Avg Steps |
|------------------|-----------|---|-----------|
| Herbivore | 3 | 50% | 156,746 |
| Aquatic | 2 | 33% | 35,000 |
| Predator | 1 | 17% | 50,000 |

---

## Conclusioni

### Strategie Vincenti

1. **Herbivore + Brain Complesso**: Il lineage #269 dimostra che cervelli complessi (24+ layers) combinati con strategia non-predatoria producono il miglior successo evolutivo.

2. **Trade-off Predazione**: I predatori hanno cervelli +21% piu complessi ma vivono -55% meno. La predazione e una strategia ad alto rischio.

3. **Selezione Naturale Forte**: Il 99.6% dei lineages si estingue. Solo i piu adatti sopravvivono.

4. **Fame > Predazione**: Il 62% delle morti e per fame, solo 3.5% per predazione. La competizione per le risorse e piu importante della predazione diretta.

### Metriche Chiave Raggiunte

- Brain medio: **30.0 layers** (massimo configurato)
- Generazione massima: **342**
- Evoluzione stabile con 3 lineages sopravvissuti
- Equilibrio ecosistemico con ~92% herbivore, ~8% predator

---

*Report generato automaticamente da PRIMORDIAL V2 Analysis System*
