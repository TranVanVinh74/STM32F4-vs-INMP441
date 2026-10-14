# Real-Time Sound Source Localization (DOA) & TinyML Audio Classifier on STM32F411

An end-to-end edge acoustic sensing system implemented on an ARM Cortex-M4 microcontroller. The system synchronizes a 4-microphone array via dual I2S/DMA, computes real-time Time Difference of Arrival (TDOA) using closed-form analytical phase solutions in the frequency domain, and filters environmental noise (fan, keyboard) from target speech using an on-device TinyML Multi-Layer Perceptron (MLP).

![Hardware Prototype](Images/H2.png)
*(Replace with your hardware prototype image or TFT radar display screenshot)*

---

## Key Features & Highlights

- **Dual I2S Synchronous Acquisition:** Captures 4-channel 24-bit audio at 16 kHz using two I2S peripherals sharing SCK/WS clocks, managed by circular DMA with double buffering (ping-pong).
- **Fast Closed-Form TDOA:** Exploits geometric symmetry of a square array to solve source angles analytically per frequency bin via `atan2`, completely avoiding full-space grid searches.
- **On-Device TinyML Classifier:** Compact MLP (1,729 parameters) running on-chip to discriminate target speech/music (`KEEP`) from acoustic noise (`NOISE`).
- **Source Tracking & Hysteresis:** Implements an angular tracker with dual-threshold hysteresis ($0.6 / 0.4$) to eliminate display jitter and false transitions.
- **Real-Time Display:** Visualizes azimuth radar sweeps and detected source directions on an ST7735 TFT display via SPI.

---

## Performance & Resource Utilization

Measured on **STM32F411CEU6** (100 MHz Cortex-M4F, 128 KiB SRAM, 512 KiB Flash):

| Metric | Measured Value | Constraint / Limit |
| :--- | :--- | :--- |
| **Angular Accuracy (1 Source, 0.1–0.3m)** | **MAE < 3.9°** | — |
| **Execution Latency per Hop** | **10.0 – 11.1 ms** | 16.0 ms budget (50% overlap) |
| **Real-Time CPU Headroom** | **~30% Idle** | Hard real-time (0 frame drops) |
| **Flash Memory Usage** | **140.70 KiB (27.5%)** | 512 KiB available |
| **SRAM Memory Usage** | **42.92 KiB (33.5%)** | 128 KiB available |
| **Average Power Consumption** | **355 mW (71 mA @ 5V)** | USB powered |

---

## System Architecture & Pipeline

```text
4x INMP441 Mics ──> Dual I2S (DMA) ──> Ping-Pong Buffer (512 samples/ch)
                                                │
                                                ▼
                                     CMSIS-DSP RFFT (512-pt)
                                                │
                                                ▼
                                    Phase Difference per Bin
                                                │
                                                ▼
                                  Closed-Form Azimuth Extraction
                                                │
                                                ▼
                                    Histogram Accumulation & NMS
                                                │
                                                ▼
                                     16-Band Spectral Energy
                                                │
                                                ▼
                                      TinyML MLP Classifier
                                                │
                                                ▼
                                   ST7735 TFT Radar / UART Log
