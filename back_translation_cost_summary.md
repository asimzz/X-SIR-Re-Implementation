# Back Translation Cost Analysis Summary

## Overview

This document summarizes the token costs for back translation experiments across 10 target languages using two different language models: **llama-3.2-1B** and **aya-23-8B**. Each experiment translates 500 input samples and generates 5,000 output samples through back-translation to 10 different intermediate languages.

## llama-3.2-1B Model Results

| Target Language | Input Tokens | Output Tokens | Total Tokens | Cost Multiplier | Input Samples | Output Samples |
|----------------|--------------|---------------|--------------|-----------------|---------------|----------------|
| cs (Czech)     | 146,223      | 1,410,458     | 1,556,681    | 9.65           | 500           | 5,000          |
| hr (Croatian)  | 174,620      | 1,407,826     | 1,582,446    | 8.06           | 500           | 5,000          |
| nl (Dutch)     | 150,108      | 1,454,421     | 1,604,529    | 9.69           | 500           | 5,000          |
| ko (Korean)    | 136,535      | 1,321,750     | 1,458,285    | 9.68           | 500           | 5,000          |
| it (Italian)   | 148,918      | 1,459,008     | 1,607,926    | 9.80           | 500           | 5,000          |
| da (Danish)    | 153,881      | 1,445,947     | 1,599,828    | 9.40           | 500           | 5,000          |
| pt (Portuguese)| 141,541      | 1,446,075     | 1,587,616    | 10.22          | 500           | 5,000          |
| ar (Arabic)    | 152,172      | 1,413,234     | 1,565,406    | 9.29           | 500           | 5,000          |
| pl (Polish)    | 175,595      | 1,396,925     | 1,572,520    | 7.96           | 500           | 5,000          |
| es (Spanish)   | 139,413      | 1,461,316     | 1,600,729    | 10.48          | 500           | 5,000          |

## aya-23-8B Model Results

| Target Language | Input Tokens | Output Tokens | Total Tokens | Cost Multiplier | Input Samples | Output Samples |
|----------------|--------------|---------------|--------------|-----------------|---------------|----------------|
| cs (Czech)     | 124,095      | 1,171,188     | 1,295,283    | 9.44           | 500           | 5,000          |
| hr (Croatian)  | 154,497      | 1,171,425     | 1,325,922    | 7.58           | 500           | 5,000          |
| nl (Dutch)     | 118,665      | 1,221,927     | 1,340,592    | 10.30          | 500           | 5,000          |
| it (Italian)   | 118,790      | 1,225,020     | 1,343,810    | 10.31          | 500           | 5,000          |
| da (Danish)    | 144,048      | 1,183,105     | 1,327,153    | 8.21           | 500           | 5,000          |
| pt (Portuguese)| 111,544      | 1,206,516     | 1,318,060    | 10.82          | 500           | 5,000          |
| ko (Korean)    | 114,239      | 1,101,842     | 1,216,081    | 9.65           | 500           | 5,000          |
| ar (Arabic)    | 128,397      | 1,154,912     | 1,283,309    | 8.99           | 500           | 5,000          |
| pl (Polish)    | 124,305      | 1,179,740     | 1,304,045    | 9.49           | 500           | 5,000          |
| es (Spanish)   | 114,101      | 1,214,712     | 1,328,813    | 10.65          | 500           | 5,000          |

## Comparative Analysis

### Model Efficiency Comparison

| Model        | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens | Avg Cost Multiplier |
|-------------|------------------|-------------------|------------------|-------------------|
| llama-3.2-1B | 151,801         | 1,431,696         | 1,583,497        | 9.34              |
| aya-23-8B    | 122,768         | 1,183,046         | 1,305,814        | 9.54              |

### Key Findings

1. **Token Efficiency**: aya-23-8B demonstrates superior token efficiency, using approximately **19% fewer input tokens** and **17% fewer output tokens** compared to llama-3.2-1B.

2. **Cost Multiplier Analysis**:
   - **llama-3.2-1B**: Range from 7.96 (Polish) to 10.48 (Spanish)
   - **aya-23-8B**: Range from 7.58 (Croatian) to 10.82 (Portuguese)

3. **Language-Specific Patterns**:
   - **Most Expensive**: Portuguese and Spanish consistently show higher cost multipliers
   - **Most Efficient**: Croatian and Polish tend to have lower cost multipliers
   - **Consistent Performance**: Korean shows stable efficiency across both models

4. **Translation Volume**: Each experiment consistently generates **10 back-translation files** per target language, maintaining a **10:1 output-to-input sample ratio**.

## Technical Details

- **Input Samples**: 500 per experiment
- **Output Samples**: 5,000 per experiment (10x multiplication)
- **Back-translation Files**: 10 per target language
- **Methodology**: KGW (seed 0) watermarking technique
- **Dataset**: MC4 English corpus