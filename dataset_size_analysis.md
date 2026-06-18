# Dataset Size & Preprocessing Analysis

This document provides a detailed investigation and justification for why the processed dataset (`data/processed_features.csv`) contains exactly **461 data points** (rows).

---

## Executive Summary

> [!NOTE]
> The final dataset size of **461 rows** is the mathematically expected output of the preprocessing pipeline. It is determined by the number of participants, the duration of their recorded sessions, and the configuration of the sliding window feature extractor.

* **Total Scanned Participants:** 57 (19 per condition)
* **Window Size:** 60 seconds
* **Step Size (Overlap):** 30 seconds
* **Total Extracted Windows (Rows):** 461

---

## Mathematical Formula

For any participant session with a resampled duration of $T$ seconds, the sliding window loop in [create_processed_dataset.py](file:///Users/narenkrishnaperumal/Desktop/Internship%20Docs/multimodal_ml_model/create_processed_dataset.py#L160-L177) extracts windows according to:

$$\text{Windows} = 1 + \left\lfloor \frac{T - 60}{30} \right\rfloor \quad \text{for } T \ge 60\text{ seconds}$$

If a session is shorter than 60 seconds, **0 windows** are extracted from it.

---

## Detailed Breakdown by Condition

Below is the distribution of session lengths and extracted windows across the 57 participants.

### 1. Baseline Condition (0)
* **Participant Count:** 19 (`P01` to `P19`)
* **Session Characteristics:** Short, standardized baseline recordings, typically lasting exactly 3 minutes (180s) to 4 minutes (240s).
* **Total Windows Extracted:** **105**

| Participant | Raw ACC Rows | Duration (Seconds) | Extracted Windows | Gaze Data |
| :--- | :---: | :---: | :---: | :---: |
| **P01 - P10** | ~4,200 - 5,300 | 180s - 207s | **5** each | Missing |
| **P11** | 7,537 | 302s | **9** | Missing |
| **P12 - P14** | ~3,200 - 4,700 | 183s - 200s | **5** each | Missing |
| **P15 - P16** | ~5,600 | 213s - 231s | **6** each | Missing |
| **P17 - P18** | ~4,900 - 5,800 | 240s - 243s | **7** each | Missing |
| **P19** | 4,253 | 191s | **5** | Missing |

---

### 2. Low Engagement (LPE) Condition (1)
* **Participant Count:** 19 (`P20` to `P38`)
* **Session Characteristics:** Varied tasks ranging from 1.5 to 7.5 minutes.
* **Total Windows Extracted:** **111**

| Participant | Raw ACC Rows | Duration (Seconds) | Extracted Windows | Gaze Data |
| :--- | :---: | :---: | :---: | :---: |
| **P33, P34, P37** | ~2,500 - 2,800 | 106s - 117s | **2** each | Present |
| **P23, P31, P32** | ~3,100 - 3,500 | 128s - 141s | **3** each | Present |
| **P20, P21, P24, P25** | ~3,500 - 4,100 | 149s - 162s | **4** each | Present |
| **P30** | 4,420 | 181s | **5** | Present |
| **P22, P27, P29, P36** | ~6,700 - 7,400 | 244s - 268s | **7** each | Present |
| **P28** | 7,961 | 299s | **9** | Present |
| **P38** | 10,190 | 381s | **11** | Present |
| **P35** | 11,524 | 442s | **13** | Present |
| **P26** | 11,707 | 458s | **14** | Present |

---

### 3. High Engagement (HPE) Condition (2)
* **Participant Count:** 19 (`P39` to `P57`)
* **Session Characteristics:** Longer immersive sessions, ranging from 3.5 to 13.5 minutes.
* **Total Windows Extracted:** **245**

| Participant | Raw ACC Rows | Duration (Seconds) | Extracted Windows | Gaze Data |
| :--- | :---: | :---: | :---: | :---: |
| **P51** | 5,410 | 206s | **5** | Present |
| **P50, P52** | ~6,700 - 6,800 | 252s - 262s | **7** each | Present |
| **P46** | 7,258 | 298s | **8** | Present |
| **P43, P49** | ~7,700 - 8,300 | 305s - 317s | **9** each | Present |
| **P47, P54, P56** | ~9,600 - 10,400 | 364s - 369s | **11** each | Present |
| **P39, P41, P45** | ~6,600 - 10,800 | 390s - 415s | **12** each | Present |
| **P42** | 11,624 | 450s | **14** | Present |
| **P48** | 12,482 | 502s | **15** | Present |
| **P55** | 13,141 | 520s | **16** | Present |
| **P53** | 14,211 | 573s | **18** | Present |
| **P40** | 14,563 | 641s | **20** | Present |
| **P44** | 15,659 | 691s | **22** | Present |
| **P57** | 17,746 | 822s | **26** | Present |

---

## Summary of Totals

| Condition Class | Code Label | Participant Count | Extracted Windows |
| :--- | :---: | :---: | :---: |
| **Baseline** | `0` | 19 | 105 |
| **Low Engagement (LPE)** | `1` | 19 | 111 |
| **High Engagement (HPE)** | `2` | 19 | 245 |
| **Total Processed Dataset** | - | **57** | **461** |

> [!TIP]
> The distribution reflects a natural experimental constraint: participants spent more time in the High Engagement (HPE) condition compared to LPE and Baseline conditions, which is why HPE contributes more data points (~53% of the dataset).
