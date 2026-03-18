# Analisi Simulazione PRIMORDIAL V2
**Data**: 2026-01-30
**Step Simulazione**: 99,950
**Configurazione**: Pre-Climate System (vecchia impostazione)

---

## Stato Generale Simulazione

| Metrica | Valore |
|---------|--------|
| Organismi nati totali | 1,194,301 |
| Attualmente vivi | 105,454 |
| Morti totali | 1,088,847 |
| Generazione massima | 81 |
| Generazione media (vivi) | 8.7 |

---

## Analisi Cognitiva - Complessità Cerebrale

### Distribuzione Cervelli (Organismi Vivi)

| Brain Layers | Count | Avg Neurons | Avg Connections |
|--------------|-------|-------------|-----------------|
| 0 | 60,197 | 0.0 | 1,439.9 |
| 1 | 1,159 | 4.0 | 457.2 |
| 2 | 62 | 9.0 | 535.6 |
| 3 | 31 | 12.8 | 528.8 |
| 4 | 36 | 16.8 | 558.4 |
| 5 | 105 | 20.1 | 501.0 |
| 6 | 535 | 24.2 | 530.7 |
| 7 | 4,264 | 27.3 | 518.2 |
| 8 | 5,696 | 31.2 | 539.8 |
| 9 | 5,057 | 35.4 | 574.1 |
| 10 | 4,829 | 39.5 | 603.2 |
| 11 | 4,003 | 43.6 | 629.2 |
| 12 | 3,041 | 47.7 | 658.2 |
| 13 | 2,054 | 51.5 | 684.5 |
| 14 | 1,371 | 55.4 | 698.6 |
| 15 | 1,226 | 59.6 | 710.5 |
| 16 | 1,448 | 63.6 | 719.6 |
| 17 | 1,539 | 67.2 | 734.1 |
| 18 | 1,554 | 71.3 | 751.3 |
| 19 | 1,320 | 75.3 | 774.4 |
| 20 | 1,016 | 79.6 | 797.3 |
| 21 | 716 | 83.7 | 819.5 |
| 22 | 552 | 87.1 | 823.5 |
| 23 | 414 | 90.3 | 833.3 |
| 24 | 330 | 94.6 | 834.6 |
| 25 | 221 | 97.6 | 819.1 |
| 26 | 147 | 100.8 | 849.0 |
| 27 | 135 | 106.3 | 952.8 |
| 28 | 189 | 110.1 | 1,013.6 |
| 29 | 267 | 114.7 | 1,084.5 |
| 30 | 1,940 | 120.5 | 1,137.1 |

### Osservazioni
- **57% degli organismi vivi** ha 0 hidden layers (cervelli minimi)
- **Distribuzione bimodale**: picco a 0 layer e plateau tra 7-12 layer
- I cervelli più complessi (30 layer) hanno fino a **125 neuroni** e **1137 connessioni**

---

## Correlazione Complessità Cerebrale vs Sopravvivenza

| Brain Layers | Count Totale | Avg Lifespan (steps) | Avg Generation |
|--------------|--------------|----------------------|----------------|
| 0 | 125,481 | 24,038 | 0.8 |
| 1 | 63,385 | 994 | 2.9 |
| 2 | 33,831 | 191 | 4.9 |
| 3 | 23,094 | 187 | 6.3 |
| 4 | 19,702 | 237 | 7.4 |
| 5 | 21,315 | 431 | 8.4 |
| 6 | 30,601 | 1,119 | 9.4 |
| 7 | 103,734 | 2,942 | 10.6 |
| 8 | 178,585 | 2,682 | 11.3 |
| 9 | 154,126 | 3,046 | 12.4 |
| 10 | 101,893 | 4,206 | 13.9 |
| 11 | 69,904 | 5,069 | 15.8 |
| 12 | 50,038 | 5,526 | 17.6 |
| 13 | 38,570 | 5,384 | 19.7 |
| 14 | 31,968 | 5,019 | 21.6 |
| 15 | 26,642 | 5,263 | 23.3 |
| 16 | 22,968 | 6,101 | 24.6 |
| 17 | 18,103 | **7,110** | 25.5 |
| 18 | 13,853 | **8,347** | 26.2 |
| 19 | 9,763 | **9,427** | 26.7 |
| 20 | 6,478 | **10,405** | 27.1 |
| 21 | 4,725 | 9,802 | 29.0 |
| 22 | 3,522 | 9,736 | 31.1 |
| 23 | 2,749 | 9,035 | 33.9 |
| 24 | 2,079 | 9,053 | 36.2 |
| 25 | 1,246 | 9,793 | 38.0 |
| 26 | 880 | 9,112 | 42.2 |
| 27 | 718 | 9,913 | 46.0 |
| 28 | 1,014 | 9,579 | 54.1 |
| 29 | 2,207 | 6,219 | 62.9 |
| 30 | 31,328 | 3,237 | 69.8 |

### Insight Chiave
- **Sweet spot**: 17-20 layer con durata media **7,000-10,400 step**
- Layer 0: sopravvivenza alta (24K) ma generazioni basse - semplici ma efficienti
- Layer 30: sopravvivenza ridotta (3.2K) - costo metabolico troppo alto
- **Trade-off chiaro**: complessità vs efficienza energetica

---

## Apprendimento Hebbian

### Statistiche Generali

| Metrica | Valore |
|---------|--------|
| Eventi di apprendimento totali | 32,978,396 |
| Organismi con apprendimento | 432,971 |
| Reward medio totale | 9.03 |
| Successi medi per organismo | 1,017.5 |
| Fallimenti medi per organismo | 456.6 |
| Max aggiornamenti pesi | 154,969 |
| Media aggiornamenti pesi | 24,783 |

### Performance per Complessità Cerebrale

| Brain Layers | Organismi | Avg Reward | Avg Success | Avg Fail | Avg Updates | Success Ratio |
|--------------|-----------|------------|-------------|----------|-------------|---------------|
| 0 | 26,345 | **-0.66** | 7.1 | 69.5 | 77 | 0.10 |
| 1 | 33,828 | -0.29 | 10.7 | 77.6 | 177 | 0.14 |
| 2 | 23,927 | -0.09 | 26.8 | 92.9 | 361 | 0.29 |
| 3 | 18,188 | 0.17 | 55.6 | 93.6 | 603 | 0.59 |
| 4 | 16,164 | 0.95 | 96.7 | 81.1 | 895 | 1.19 |
| 5 | 17,791 | 1.66 | 149.8 | 93.7 | 1,474 | 1.60 |
| 6 | 24,944 | 2.48 | 255.0 | 119.5 | 2,654 | 2.13 |
| 7 | 78,644 | 8.46 | 1,033.6 | 458.1 | 15,196 | 2.26 |
| 8 | 123,032 | 8.80 | 984.7 | 527.9 | 18,933 | 1.87 |
| 9 | 117,878 | 9.09 | 1,067.4 | 441.6 | 21,972 | 2.42 |
| 10 | 85,961 | 9.46 | 1,058.7 | 456.3 | 24,738 | 2.32 |
| 11 | 62,594 | 9.56 | 1,069.1 | 459.0 | 27,138 | 2.33 |
| 12 | 46,794 | 9.67 | 1,062.4 | 457.0 | 29,519 | 2.32 |
| 13 | 37,300 | 9.73 | 1,069.8 | 463.3 | 31,984 | 2.31 |
| 14 | 31,372 | 9.88 | 1,065.3 | 469.4 | 34,841 | 2.27 |
| 15 | 26,176 | 9.96 | 1,062.7 | 472.4 | 37,592 | 2.25 |
| 16 | 22,568 | **10.00** | 1,068.5 | 469.7 | 40,101 | 2.27 |
| 17 | 17,708 | 10.02 | 1,057.8 | 470.2 | 42,412 | 2.25 |
| 18 | 13,547 | 9.97 | 1,061.8 | 468.8 | 44,496 | 2.26 |
| 19 | 9,577 | 9.92 | 1,053.9 | 460.0 | 46,354 | 2.29 |
| 20 | 6,381 | 9.82 | 1,060.2 | 453.3 | 48,262 | 2.34 |
| 21 | 4,627 | 9.94 | 1,039.1 | 453.0 | 50,256 | 2.29 |
| 22 | 3,462 | 9.77 | 1,036.6 | 443.2 | 51,463 | 2.34 |
| 23 | 2,695 | 10.03 | 1,026.7 | 437.3 | 53,059 | 2.35 |
| 24 | 2,037 | 9.75 | 978.7 | 412.0 | 53,359 | 2.38 |
| 25 | 1,216 | 9.24 | 1,002.7 | 389.5 | 52,441 | 2.57 |
| 26 | 855 | 9.21 | 1,002.3 | 375.3 | 53,630 | 2.67 |
| 27 | 695 | 8.98 | 930.0 | 379.3 | 50,247 | 2.45 |
| 28 | 982 | 12.74 | 843.7 | 198.4 | 42,668 | 4.25 |
| 29 | 2,184 | **19.69** | 605.7 | 132.6 | 25,022 | 4.57 |
| 30 | 30,944 | 10.06 | 380.6 | 60.5 | 14,719 | **6.29** |

### Insight Apprendimento
1. **Layer 0 ha reward negativo** (-0.66) - non impara efficacemente
2. **Plateau di performance** tra 7-20 layer (reward ~9-10)
3. **Layer 28-30 sono più selettivi**: meno successi totali ma ratio 3-6x migliore
4. I cervelli complessi "scelgono meglio" quando agire

### Top 10 Learner (Organismi Vivi)

| Organism ID | Brain Layers | Generation | Total Reward | Successes | Failures | Success Ratio |
|-------------|--------------|------------|--------------|-----------|----------|---------------|
| 40852 | 30 | 71 | 53.22 | 4,950 | 1 | **4,950.00** |
| 40856 | 30 | 72 | 53.22 | 4,949 | 1 | 4,949.00 |
| 40897 | 30 | 76 | 53.06 | 4,937 | 1 | 4,937.00 |
| 42619 | 30 | 78 | 70.42 | 4,780 | 1 | 4,780.00 |
| 1937 | 9 | 13 | 55.22 | 4,254 | 1 | 4,254.00 |
| 1990 | 9 | 15 | 55.61 | 4,251 | 1 | 4,251.00 |
| 2096 | 14 | 20 | 55.88 | 4,244 | 1 | 4,244.00 |
| 2151 | 17 | 23 | 55.80 | 4,241 | 1 | 4,241.00 |
| 2588 | 9 | 16 | 54.31 | 4,204 | 1 | 4,204.00 |
| 2759 | 10 | 18 | 54.76 | 4,187 | 1 | 4,187.00 |

---

## Predatori vs Non-Predatori

| Tipo | Totale | Vivi | % Vivi | Avg Layers | Max Layers | Avg Neurons | Avg Gen |
|------|--------|------|--------|------------|------------|-------------|---------|
| Non-predatori | 1,080,093 | 100,884 | 9.3% | 8.52 | 30 | 33.7 | 13.7 |
| Predatori | 114,409 | 4,570 | 4.0% | **11.23** | 30 | **43.8** | 14.2 |

### Osservazioni
- I predatori sono il **4.3%** della popolazione viva
- Hanno cervelli **32% più complessi** (11.2 vs 8.5 layer)
- **Neuroni 30% in più** rispetto ai non-predatori
- Tasso di sopravvivenza inferiore (4% vs 9.3%)

---

## Evoluzione Temporale della Complessità

| Step Range | Nati | Avg Layers | Max Layers | Avg Neurons |
|------------|------|------------|------------|-------------|
| 0-10,000 | 707,499 | 5.56 | 30 | 21.9 |
| 10,000-20,000 | 132,623 | 10.08 | 30 | 39.6 |
| 20,000-30,000 | 80,930 | 11.14 | 30 | 44.0 |
| 30,000-40,000 | 64,240 | 12.05 | 30 | 47.7 |
| 40,000-50,000 | 34,824 | 13.66 | 30 | 54.1 |
| 50,000-60,000 | 53,969 | **20.28** | 30 | **81.4** |
| 60,000-70,000 | 30,843 | 14.76 | 25 | 58.5 |
| 70,000-80,000 | 30,935 | 15.80 | 27 | 62.5 |
| 80,000-90,000 | 29,471 | 16.91 | 29 | 66.9 |
| 90,000-100,000 | 29,168 | 17.96 | 30 | 71.2 |

### Trend Evolutivo
- **Crescita costante** della complessità: da 5.5 a 18 layer
- **Picco anomalo** a step 50K-60K (20.28 layer) - possibile evento selettivo
- La complessità media è aumentata del **223%** durante la simulazione

---

## Lineage più Prolifici

| Lineage ID | Discendenti Totali | Vivi | Max Gen | Avg Layers |
|------------|-------------------|------|---------|------------|
| **269** | 53,720 | 3,051 | **81** | **24.1** |
| 1315 | 37,921 | 843 | 31 | 14.2 |
| 765 | 35,000 | 690 | 32 | 10.7 |
| 1441 | 32,801 | 575 | 31 | 10.4 |
| 1944 | 32,335 | 1,086 | 20 | 8.8 |
| 1634 | 31,807 | 757 | 32 | 11.3 |
| 4558 | 26,030 | 569 | 31 | 11.0 |
| 2582 | 25,214 | 549 | 30 | 11.2 |
| 3153 | 22,602 | 827 | 19 | 9.4 |
| 3256 | 20,139 | 404 | 31 | 11.7 |

### Lineage Dominante: #269
- **53,720 discendenti** (4.5% di tutti gli organismi mai nati)
- **3,051 vivi** (2.9% della popolazione attuale)
- **Generazione massima 81** - il lignaggio più evoluto
- **Layer medi 24.1** - cervelli molto complessi

---

## Comportamento Attuale (ultimi 1000 step)

| Azione | Count | Percentuale |
|--------|-------|-------------|
| Eat | 99,444 | **99.0%** |
| (nessuna) | 684 | 0.7% |
| Attack | 133 | 0.1% |
| Reproduce | 32 | 0.0% |
| MoveSouth | 21 | 0.0% |
| SignalDanger | 18 | 0.0% |
| MoveEast | 18 | 0.0% |
| Altri movimenti | ~80 | 0.1% |

### Osservazione
La popolazione è in una **fase di "grazing"** con 99% delle azioni dedicate al consumo di cibo. Pochi movimenti e riproduzione suggeriscono:
- Abbondanza di cibo nella posizione corrente
- Popolazione vicina alla carrying capacity
- Pressione selettiva bassa

---

## Conclusioni

### Punti Chiave
1. **Evoluzione cognitiva evidente**: complessità media triplicata in 100K step
2. **Sweet spot a 17-20 layer**: miglior rapporto costo/beneficio
3. **Apprendimento Hebbian funziona**: reward positivo da 7+ layer
4. **I cervelli complessi sono selettivi**: meno azioni ma più efficaci
5. **Lineage #269 domina**: famiglia più evoluta e prolifica
6. **Predatori più intelligenti**: 32% più layer ma meno numerosi

### Suggerimenti per Prossime Analisi
- Monitorare il comportamento dopo l'attivazione del sistema climatico
- Analizzare come temperatura/umidità influenzano la distribuzione spaziale
- Verificare se il clima favorisce cervelli più complessi (termoregolazione)
- Tracciare l'adattamento dei lineage ai diversi biomi

---

*Generato automaticamente da Claude Code - 2026-01-30*
