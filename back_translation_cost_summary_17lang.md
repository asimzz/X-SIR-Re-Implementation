# Back Translation Cost Analysis Summary - 17 Languages

## Overview

This document summarizes the token costs for back translation experiments across 17 target languages using two different language models: **llama-3.2-1B** and **aya-23-8B**. Each experiment translates 500 input samples and generates 8,500 output samples through back-translation to 17 different intermediate languages, representing a significant expansion from the previous 10-language experiments.

The 17 languages covered include: Bengali (bn), German (de), Spanish (es), Persian (fa), French (fr), Hindi (hi), Italian (it), Hebrew (iw), Japanese (ja), Korean (ko), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Tamil (ta), Ukrainian (uk), and Vietnamese (vi).

## llama-3.2-1B Model Results

| Target Language        | Input Tokens | Output Tokens | Total Tokens | Cost Multiplier | Input Samples | Output Samples |
|------------------------|--------------|---------------|--------------|-----------------|---------------|----------------|
| bn (Bengali)           | 571,334      | 3,165,853     | 3,737,187    | 5.54           | 500           | 8,500          |
| de (German)            | 146,493      | 3,618,076     | 3,764,569    | 24.70          | 500           | 8,500          |
| es (Spanish)           | 139,411      | 3,699,506     | 3,838,917    | 26.54          | 500           | 8,500          |
| fa (Persian)           | 143,558      | 3,499,788     | 3,643,346    | 24.38          | 500           | 8,500          |
| fr (French)            | 146,645      | 3,688,484     | 3,835,129    | 25.15          | 500           | 8,500          |
| hi (Hindi)             | 242,852      | 3,525,199     | 3,768,051    | 14.52          | 500           | 8,500          |
| it (Italian)           | 148,924      | 3,708,339     | 3,857,263    | 24.90          | 500           | 8,500          |
| iw (Hebrew)            | 344,049      | 3,346,328     | 3,690,377    | 9.73           | 500           | 8,500          |
| ja (Japanese)          | 148,771      | 3,523,864     | 3,672,635    | 23.69          | 500           | 8,500          |
| ko (Korean)            | 136,496      | 3,351,197     | 3,487,693    | 24.55          | 500           | 8,500          |
| nl (Dutch)             | 150,274      | 3,699,184     | 3,849,458    | 24.62          | 500           | 8,500          |
| pl (Polish)            | 175,453      | 3,593,095     | 3,768,548    | 20.48          | 500           | 8,500          |
| pt (Portuguese)        | 141,560      | 3,659,337     | 3,800,897    | 25.85          | 500           | 8,500          |
| ru (Russian)           | 154,579      | 3,685,450     | 3,840,029    | 23.84          | 500           | 8,500          |
| ta (Tamil)             | 701,400      | 2,894,952     | 3,596,352    | 4.13           | 500           | 8,500          |
| uk (Ukrainian)         | 158,958      | 3,552,755     | 3,711,713    | 22.35          | 500           | 8,500          |
| vi (Vietnamese)        | 135,089      | 3,552,555     | 3,687,644    | 26.30          | 500           | 8,500          |

## aya-23-8B Model Results

| Target Language        | Input Tokens | Output Tokens | Total Tokens | Cost Multiplier | Input Samples | Output Samples |
|------------------------|--------------|---------------|--------------|-----------------|---------------|----------------|
| bn (Bengali)           | 467,729      | 2,423,877     | 2,891,606    | 5.18           | 500           | 8,500          |
| de (German)            | 119,441      | 2,751,028     | 2,870,469    | 23.03          | 500           | 8,500          |
| es (Spanish)           | 114,169      | 2,851,446     | 2,965,615    | 24.98          | 500           | 8,500          |
| fa (Persian)           | 129,243      | 2,691,085     | 2,820,328    | 20.82          | 500           | 8,500          |
| fr (French)            | 121,038      | 2,843,150     | 2,964,188    | 23.49          | 500           | 8,500          |
| hi (Hindi)             | 264,110      | 2,638,226     | 2,902,336    | 9.99           | 500           | 8,500          |
| it (Italian)           | 118,690      | 2,879,076     | 2,997,766    | 24.26          | 500           | 8,500          |
| iw (Hebrew)            | 130,282      | 2,689,447     | 2,819,729    | 20.64          | 500           | 8,500          |
| ja (Japanese)          | 111,935      | 2,711,960     | 2,823,895    | 24.23          | 500           | 8,500          |
| ko (Korean)            | 114,132      | 2,588,445     | 2,702,577    | 22.68          | 500           | 8,500          |
| nl (Dutch)             | 118,608      | 2,878,800     | 2,997,408    | 24.27          | 500           | 8,500          |
| pl (Polish)            | 124,281      | 2,786,025     | 2,910,306    | 22.42          | 500           | 8,500          |
| pt (Portuguese)        | 111,501      | 2,820,991     | 2,932,492    | 25.30          | 500           | 8,500          |
| ru (Russian)           | 120,120      | 2,845,568     | 2,965,688    | 23.69          | 500           | 8,500          |
| ta (Tamil)             | 492,253      | 2,249,816     | 2,742,069    | 4.57           | 500           | 8,500          |
| uk (Ukrainian)         | 127,806      | 2,720,485     | 2,848,291    | 21.29          | 500           | 8,500          |
| vi (Vietnamese)        | 120,606      | 2,689,607     | 2,810,213    | 22.30          | 500           | 8,500          |

## Comparative Analysis

### Model Efficiency Comparison

| Model        | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens | Avg Cost Multiplier |
|-------------|------------------|-------------------|------------------|-------------------|
| llama-3.2-1B | 222,696         | 3,515,527         | 3,738,224        | 20.66              |
| aya-23-8B    | 170,937         | 2,709,354         | 2,880,292        | 20.18              |

### Key Findings

1. **Token Efficiency**: aya-23-8B demonstrates superior token efficiency, using approximately **23% fewer input tokens** and **23% fewer output tokens** compared to llama-3.2-1B.

2. **Cost Multiplier Analysis**:
   - **llama-3.2-1B**: Range from 4.13 (Tamil) to 26.54 (Spanish)
   - **aya-23-8B**: Range from 4.57 (Tamil) to 25.30 (Portuguese)

3. **Language-Specific Patterns**:
   - **Most Expensive**: Spanish, Vietnamese, and Portuguese consistently show the highest cost multipliers
   - **Most Efficient**: Tamil, Bengali, and Hindi show significantly lower cost multipliers due to higher input token requirements
   - **Consistent High-Cost**: European languages (German, French, Italian, Dutch) generally require higher cost multipliers

4. **Scale Comparison with 10-Language Experiments**:
   - **17-Language Setup**: 17 back-translation files per target language, 8,500 output samples per experiment
   - **10-Language Setup**: 10 back-translation files per target language, 5,000 output samples per experiment
   - **Scale Factor**: 70% increase in back-translation files, 70% increase in output samples

5. **Resource-Intensive Languages**:
   - **Tamil** and **Bengali** require significantly more input tokens but result in lower cost multipliers
   - **Hebrew** shows moderate input token requirements with reasonable cost efficiency
   - **Hindi** demonstrates balanced performance across both models

### Efficiency Improvements in 17-Language Setup

Compared to the previous 10-language experiments:

- **Higher throughput**: 8,500 vs 5,000 output samples per experiment
- **Broader language coverage**: 17 vs 10 intermediate languages
- **Maintained efficiency**: Similar cost multiplier ranges despite increased complexity

## Technical Details

- **Input Samples**: 500 per experiment
- **Output Samples**: 8,500 per experiment (17x multiplication)
- **Back-translation Files**: 17 per target language
- **Methodology**: KGW (seed 0) watermarking technique
- **Dataset**: MC4 English corpus
- **Language Coverage**: 17 languages spanning multiple language families and scripts
- **Expanded Scope**: Significant increase from 10-language to 17-language setup for more comprehensive multilingual evaluation
