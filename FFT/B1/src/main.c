#include "stm32f4xx.h"
#include "SysTick.h"
#include "UART.h"
#include "I2S.h"
#include "SPI.h"
#include "TFT.h"
#include "FONT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 

#include "arm_math.h"

// --- CAU HINH AUDIO & DSP ---
#define VOLUME_BOOST 1          
#define FFT_SIZE 512  
#define PI 3.14159265358979f    
#define SAMPLE_RATE 16000.0f 

// --- CAU HINH RADAR TFT ---
#define RADAR_X 64         
#define RADAR_Y 80         
#define RADAR_R 60         

// Khai bao mang FFT & Xu ly tin hieu
float32_t fft_in_m1[FFT_SIZE]; float32_t fft_in_m2[FFT_SIZE]; 
float32_t fft_in_m3[FFT_SIZE]; float32_t fft_in_m4[FFT_SIZE]; 
float32_t win_in_m1[FFT_SIZE]; float32_t win_in_m2[FFT_SIZE]; 
float32_t win_in_m3[FFT_SIZE]; float32_t win_in_m4[FFT_SIZE];
float32_t fft_out_m1[FFT_SIZE]; float32_t fft_out_m2[FFT_SIZE];
float32_t fft_out_m3[FFT_SIZE]; float32_t fft_out_m4[FFT_SIZE];

/*
+ radar_histogram: hom phieu tuc thoi la buc anh chup nhanh du lieu tho trong 1 khung hinh FFT. Dau moi khung hinh no se reset ve 0.
khi vong lap tan so chay, cac tan so se tinh ra goc nao se nem diem so (magnitude * ScorePV) va dung vi tri goc do trong mang.
tac dung: gom tat ca la phieu cua dai tan so trong khoang 32 ms.
+ accum_hist[360] bo nho dem thoi gian: la bo loc muot theo thoi gian. Mang nay dong vai tro nhu 1 tu dien tich tru nang luong.
+ smoothed_hist bo loc lam muot khong gian: lay du lieu tu accum_hist va chay thuat toan loc trung binh trong so 5 diem lan can.
*/
float32_t radar_histogram[360];
float32_t accum_hist[360]; 
float32_t smoothed_hist[360];
float32_t hann_window[FFT_SIZE];    

arm_rfft_fast_instance_f32 fft_handler;
volatile uint32_t float_index = 0;
volatile uint8_t data_ready = 0; 

extern uint16_t i2s2_rx_buffer[I2S_RX_BUFFER_SIZE];
extern uint16_t i2s3_rx_buffer[I2S_RX_BUFFER_SIZE];
extern volatile uint8_t i2s2_half; extern volatile uint8_t i2s2_full;
extern volatile uint8_t i2s3_half; extern volatile uint8_t i2s3_full;

// Ham phu tro: Tinh khoang cach goc ngan nhat tren duong tron
int get_angle_diff(int a1, int a2) {
    int diff = abs(a1 - a2) % 360;
    return diff > 180 ? 360 - diff : diff;
}

// Ham phu tro: Ve kim radar bang luong giac (DA DAO 180 DO)
void draw_needle(float angle_deg, uint16_t color) {
    float rad = (angle_deg + 180.0f) * PI / 180.0f;
    int x_end = RADAR_X + (int)(RADAR_R * sinf(rad));
    int y_end = RADAR_Y - (int)(RADAR_R * cosf(rad));
    drawLine(RADAR_X, RADAR_Y, x_end, y_end, color);
}

// ham doc mic 
void Extract_Audio(uint32_t start_index, uint32_t end_index) { 
    for (uint32_t i = start_index; i < end_index; i += 4) {
        int32_t amp_m1 = (int32_t)((int16_t)i2s2_rx_buffer[i]) * VOLUME_BOOST;
        if (amp_m1 > 32767) amp_m1 = 32767; else if (amp_m1 < -32768) amp_m1 = -32768;
        fft_in_m1[float_index] = (float32_t)amp_m1;

        int32_t amp_m3 = (int32_t)((int16_t)i2s3_rx_buffer[i]) * VOLUME_BOOST;
        if (amp_m3 > 32767) amp_m3 = 32767; else if (amp_m3 < -32768) amp_m3 = -32768;
        fft_in_m3[float_index] = (float32_t)amp_m3;

        int32_t amp_m2 = (int32_t)((int16_t)i2s2_rx_buffer[i + 2]) * VOLUME_BOOST; 
        if (amp_m2 > 32767) amp_m2 = 32767; else if (amp_m2 < -32768) amp_m2 = -32768;
        fft_in_m2[float_index] = (float32_t)amp_m2;

        int32_t amp_m4 = (int32_t)((int16_t)i2s3_rx_buffer[i + 2]) * VOLUME_BOOST; 
        if (amp_m4 > 32767) amp_m4 = 32767; else if (amp_m4 < -32768) amp_m4 = -32768; 
        fft_in_m4[float_index] = (float32_t)amp_m4;

        float_index++;
        if (float_index >= FFT_SIZE) { float_index = 0; data_ready = 1; return; }
    }
}

// ham dong vai tro nhu 1 bo xu ly mo cuon va chuan hoa pha cho 4 micro
void unwrap_and_center_phase(float* phases, float offset, float* out_phases) {
/*
giai van de lat pha. Vi du nhu voi goc 1 do va 359 do thuc chat chi lech nhau 2 do nhung neu lay 359-1 thi ra tan 358
ham nay giai quyet van de nay bang cach cong them offset ket hop chia lay du 
    VD voi goc 1 va 359 do no se lam nhu sau:
    xet voi alpha = 0 ta co 1+0=1
    359+=359; -> chenh nhau nhieu qua
    xet voi alpha = 90 do ta co 1+90=91 do 
    mic 2: 359+90=449 mod 360=89 -> thay duoc no lech dung 2 do 
*/
    float sum = 0;
    for (int i = 0; i < 4; i++) {
        float p = phases[i] + offset;
        p = fmodf(p, 2.0f * PI); // ham fmod tra ve phan du cua phep chia so thuc 
        if (p < 0) p += 2.0f * PI;
        out_phases[i] = p;
        sum += p;
    }
    float mean = sum / 4.0f; // tinh pha trung binh cua 4 mic 
    for (int i = 0; i < 4; i++) out_phases[i] -= mean; // lay pha cua tung micro tru di gia tri trung binh de tung ra do lech cua tung mic. VD no se ghi kieu: -1 +1 -3 +3 ,..
}

int main(void) {
    SystemCoreClockUpdate(); 
    SysTick_Init();
    UART_Init(USART2,921600); 
    I2S_PLLI2S_Init();
    I2S_Init(SPI3, i2s3_rx_buffer);
    I2S_Init(SPI2, i2s2_rx_buffer);
    
    SPI1_Init_Master();
    
    arm_rfft_fast_init_f32(&fft_handler, FFT_SIZE);
    for (uint32_t i = 0; i < FFT_SIZE; i++) hann_window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (FFT_SIZE - 1)));

    float mic_X[4] = {-1.0f,  1.0f, -1.0f,  1.0f}; // dung de chieu vector luc len truc X hoac truc Y
    float mic_Y[4] = {-1.0f, -1.0f,  1.0f,  1.0f};
    float d_meter = 0.0275f; 

    memset(accum_hist, 0, sizeof(accum_hist));

    // --- KHOI TAO VA VE NEN RADAR MAN HINH TFT ---
    initTFT();
    fullDisplay(0x0000); 
    drawCircle(RADAR_X, RADAR_Y, RADAR_R, 0x07E0); 

    UART_SendString(USART2,"--- CHE DO 4 MIC | THEO DOI GIONG NOI & RADAR TFT | DEV: TRAN VAN VINH ---\r\n");

    static int active_sources = 1;

    while (1) {
        if (SPI2->SR & (1 << 6)) { volatile uint32_t tmpreg = SPI2->DR; tmpreg = SPI2->SR; (void)tmpreg; }
        if (SPI3->SR & (1 << 6)) { volatile uint32_t tmpreg = SPI3->DR; tmpreg = SPI3->SR; (void)tmpreg; }
        
        // DA XOA DOAN DOC UART (STREAMING) O DAY THEO YEU CAU 
        
        if (i2s2_half && i2s3_half) { i2s2_half = 0; i2s3_half = 0; Extract_Audio(0, I2S_RX_BUFFER_SIZE / 2); }
        if (i2s2_full && i2s3_full) { i2s2_full = 0; i2s3_full = 0; Extract_Audio(I2S_RX_BUFFER_SIZE / 2, I2S_RX_BUFFER_SIZE); }

        if (data_ready) {
            data_ready = 0; 
            
            float32_t mean1=0, mean2=0, mean3=0, mean4=0;
            for(uint32_t i=0; i<FFT_SIZE; i++) { mean1+=fft_in_m1[i]; mean2+=fft_in_m2[i]; mean3+=fft_in_m3[i]; mean4+=fft_in_m4[i]; }
            mean1/=FFT_SIZE; mean2/=FFT_SIZE; mean3/=FFT_SIZE; mean4/=FFT_SIZE;

            float32_t energy = 0;
            for(uint32_t i=0; i<FFT_SIZE; i++) {
                energy += fabsf(fft_in_m1[i] - mean1);
                win_in_m1[i] = (fft_in_m1[i] - mean1) * hann_window[i];
                win_in_m2[i] = (fft_in_m2[i] - mean2) * hann_window[i];
                win_in_m3[i] = (fft_in_m3[i] - mean3) * hann_window[i];
                win_in_m4[i] = (fft_in_m4[i] - mean4) * hann_window[i];
            }
            
            // Loc cuong do nang luong chong on nen (Nguong 150)
            if (energy/FFT_SIZE < 150.0f) { 
                for(int i=0; i<360; i++) accum_hist[i] *= 0.85f; 
                continue; 
            }

            arm_rfft_fast_f32(&fft_handler, win_in_m1, fft_out_m1, 0);
            arm_rfft_fast_f32(&fft_handler, win_in_m2, fft_out_m2, 0);
            arm_rfft_fast_f32(&fft_handler, win_in_m3, fft_out_m3, 0);
            arm_rfft_fast_f32(&fft_handler, win_in_m4, fft_out_m4, 0);

            memset(radar_histogram, 0, sizeof(radar_histogram));
            
            int k_min = (int)(300.0f * FFT_SIZE / SAMPLE_RATE);
            int k_max = (int)(1500.0f * FFT_SIZE / SAMPLE_RATE);

            for (int k = k_min; k < k_max; k++) {
                float f_hz = k * (SAMPLE_RATE / FFT_SIZE);
                float k_factor = 2.0f * PI * f_hz * d_meter / 343.2f; 
                float Sum3f = 4.0f * k_factor * k_factor; 

                float raw_phases[4];
                raw_phases[0] = atan2f(fft_out_m1[2*k+1], fft_out_m1[2*k]); // su dung ham co san atan2f de tim ra goc 
                raw_phases[1] = atan2f(fft_out_m2[2*k+1], fft_out_m2[2*k]);
                raw_phases[2] = atan2f(fft_out_m3[2*k+1], fft_out_m3[2*k]);
                raw_phases[3] = atan2f(fft_out_m4[2*k+1], fft_out_m4[2*k]);

                float power = (fft_out_m1[2*k]*fft_out_m1[2*k] + fft_out_m1[2*k+1]*fft_out_m1[2*k+1]);
                float magnitude = sqrtf(power); // lay can bac 2 de ra bien do 

                float min_Variance = 1e9;
                float best_Angle = 0;
                float best_AbsPhamic = 0;
                float offsets[4] = {0, PI/2.0f, PI, 1.5f*PI}; 
                
                for(int o = 0; o < 4; o++) {
                    float PhAF[4];
                    unwrap_and_center_phase(raw_phases, offsets[o], PhAF);

                    float Phasq = 0; // Tong binh phuong cac do lech pha, do xem do lech pha cua 4 mic cang thang the nao 
                    float PhamicR = 0;
                    float PhamicI = 0;
                    for(int i = 0; i < 4; i++) {
                        Phasq += PhAF[i] * PhAF[i];
                        PhamicR += PhAF[i] * mic_X[i]; // chieu len truc X
                        PhamicI += PhAF[i] * mic_Y[i]; // chieu len truc Y
                    }

                    // thuat toan su dung ham atan2 de suy ra goc den cua am thanh 
                    float angle_rad = atan2f(PhamicI, PhamicR); 
                    
                    // --- DAO CHIEU VA XOAY TRUC TRUC TIEP ---
                    angle_rad = -angle_rad; // Ep thuan chieu kim dong ho
                    float angle_deg = angle_rad * (180.0f / PI); // Quy doi sang Do
                    angle_deg -= 90.0f; // Quay truc phan mem di 90 do
                    
                    while (angle_deg < 0.0f) angle_deg += 360.0f;
                    while (angle_deg >= 360.0f) angle_deg -= 360.0f;
                    // ----------------------------------------
                    
                    float AbsPhamic = sqrtf(PhamicR*PhamicR + PhamicI*PhamicI); // chieu dai cua vector dong thuan 

                    float MeanPV = Phasq + Sum3f; // Tong nang luong (phuong sai) goc cua tin hieu
                    float MinPV  = MeanPV - 2.0f * k_factor * AbsPhamic; // la phan nang luong rac 
                    
                    /*
                    thuat toan tim trong cac truong hop xem truong hop nao cho ra sai so MinPV nho nhat
                    (min_Variance). Goc tuong ung voi sai so nho nhat duoc chot lam 
                    goc tot nhat (best angle) cho dai tan so nay 
                    */
                    if (MinPV < min_Variance) {
                        min_Variance = MinPV;
                        best_Angle = angle_deg; // Chu y: Da luu san angle_deg chuan hoa vao day
                        best_AbsPhamic = AbsPhamic;
                    }
                }

                if (min_Variance < 0.001f) min_Variance = 0.001f; // chan loi chia cho 0
                float MeanPV_total = min_Variance + 2.0f * k_factor * best_AbsPhamic; // tong nang luong
                float ScorePV = MeanPV_total / min_Variance; // lay tong nang luong chia cho nang luong rac, sai so lon (nguon gia) ScorePV se bi nho 
                
                if (ScorePV > 12.0f) ScorePV = 12.0f; 

                if (ScorePV > 6.0f) { // neu cai diem du tren 6
                    radar_histogram[(int)best_Angle % 360] += magnitude * ScorePV;
                }
            }

            for(int i = 0; i < 360; i++) {
                accum_hist[i] = accum_hist[i] * 0.15f + radar_histogram[i] * 0.85f; 
            }

            for(int i = 0; i < 360; i++) {
                smoothed_hist[i] = (
                    accum_hist[(i+355)%360] +
                    accum_hist[(i+356)%360] * 2.0f +
                    accum_hist[(i+357)%360] * 3.0f +
                    accum_hist[(i+358)%360] * 4.0f +
                    accum_hist[(i+359)%360] * 5.0f +
                    accum_hist[i] * 6.0f +
                    accum_hist[(i+1)%360] * 5.0f +
                    accum_hist[(i+2)%360] * 4.0f +
                    accum_hist[(i+3)%360] * 3.0f +
                    accum_hist[(i+4)%360] * 2.0f +
                    accum_hist[(i+5)%360]
                ) / 36.0f;
            }

            float hist_mean = 0;
            for(int i = 0; i < 360; i++) hist_mean += smoothed_hist[i];
            hist_mean /= 360.0f;

            float dynamic_thresh = hist_mean * 2.5f; 
            // Ha buc tuong chan day xuong 100 de bat duoc giong noi binh thuong
            if (dynamic_thresh < 100.0f) dynamic_thresh = 100.0f; 
            
            typedef struct {
                int angle; 
                float val; 
            } Peak;
            
            Peak candidates[36]; 
            int cand_count = 0;

            for(int i = 0; i < 360; i++) {
                if (smoothed_hist[i] > dynamic_thresh) {
                    int is_peak = 1;
                
                    for(int d = -30; d <= 30 ; d++) {
                        if (d == 0) continue; 
                        int idx = (i + d + 360) % 360;
                        
                        if (smoothed_hist[idx] >= smoothed_hist[i] && d > 0) {
                            is_peak = 0; 
                            break;
                        }
                        if (smoothed_hist[idx] > smoothed_hist[i] && d < 0) {
                            is_peak = 0; 
                            break;
                        }
                    }
                    if (is_peak && cand_count < 36) {
                        candidates[cand_count].angle = i;
                        candidates[cand_count].val = smoothed_hist[i];
                        cand_count++;
                    }
                }
            }

            // SAP XEP MANG (Dual nested-loop layout goc)
            for(int i = 0; i < cand_count - 1; i++) {
                for(int j = i + 1; j < cand_count; j++) {
                    if(candidates[j].val > candidates[i].val) {
                        Peak temp = candidates[i];
                        candidates[i] = candidates[j];
                        candidates[j] = temp;
                    }
                }
            }

            int peak_angles[3]; 
            int peak_count = 0;
 
            if (cand_count > 0) {
                // Dinh 1 mac dinh duoc chon (Nguon to nhat)
                peak_angles[peak_count++] = candidates[0].angle;
                float max1_val = candidates[0].val;

                float thresh_N2 = (active_sources >= 2) ? 0.30f : 0.45f;
                float thresh_N3 = (active_sources >= 3) ? 0.35f : 0.55f;
                
                int min_separation = 35; 

                // TIM NGUON 2
                int idx_n2 = -1;
                for(int i = 1; i < cand_count; i++) {
                    if (candidates[i].val > max1_val * thresh_N2 && candidates[i].val > dynamic_thresh * 2.5f) {
                        if (get_angle_diff(candidates[i].angle, peak_angles[0]) >= min_separation) {
                            peak_angles[peak_count++] = candidates[i].angle;
                            idx_n2 = i; break;
                        }
                    }
                }
                
                // TIM NGU?N 3
                if (idx_n2 != -1) {
                    for(int i = idx_n2 + 1; i < cand_count; i++) {
                        if (candidates[i].val > max1_val * thresh_N3 && candidates[i].val > dynamic_thresh * 3.0f) {
                            if (get_angle_diff(candidates[i].angle, peak_angles[0]) >= min_separation &&
                                get_angle_diff(candidates[i].angle, peak_angles[1]) >= min_separation) {
                                peak_angles[peak_count++] = candidates[i].angle; break;
                            }
                        }
                    }
                }
            }
            
            active_sources = (peak_count > 0) ? peak_count : 1; 
        
            if(peak_count > 0) {
                char buf[128];
                sprintf(buf, "Phat hien %d nguon | Goc: ", peak_count);
                for(int i=0; i<peak_count; i++){ 
                    char angle_str[16]; 
                    sprintf(angle_str, "%03d ", peak_angles[i]); 
                    strcat(buf, angle_str); 
                }
                strcat(buf, "\r\n"); 
                UART_SendString(USART2, buf);
            }

            // =========================================================
            // XU LY VE TFT - RANH DOC LAP & TOI UU GIAO TIEP SPI
            // =========================================================
            
            static int target_angles[3] = {0, 0, 0};
            static int target_timers[3] = {0, 0, 0};
            
            static int drawn_active[3] = {0, 0, 0};
            static int drawn_angles[3] = {0, 0, 0};

            // Rut thoi gian song
            for(int i = 0; i < 3; i++) {
                if (target_timers[i] > 0) target_timers[i]--;
            }

            // Dua goc moi vao cac ranh
            for (int i = 0; i < peak_count; i++) {
                int best_match = -1;
                int min_diff = 60; 
                
                for (int j = 0; j < 3; j++) {
                    if (target_timers[j] > 0) {
                        int diff = get_angle_diff(peak_angles[i], target_angles[j]);
                        if (diff < min_diff) {
                            min_diff = diff;
                            best_match = j;
                        }
                    }
                }
                
                if (best_match != -1) {
                    target_angles[best_match] = peak_angles[i];
                    target_timers[best_match] = 4; 
                } else {
                    for (int j = 0; j < 3; j++) {
                        if (target_timers[j] == 0) {
                            target_angles[j] = peak_angles[i];
                            target_timers[j] = 4;
                            break;
                        }
                    }
                }
            }

          
            int need_redraw_center = 0;
            
            for(int i = 0; i < 3; i++) {
                if (target_timers[i] > 0) {
                    if (drawn_active[i] == 0 || drawn_angles[i] != target_angles[i]) {
                        if (drawn_active[i]) {
                            draw_needle((float)drawn_angles[i], 0x0000); 
                            need_redraw_center = 1;
                        }
                        draw_needle((float)target_angles[i], 0xF800); 
                        drawn_angles[i] = target_angles[i];
                        drawn_active[i] = 1;
                    }
                } else {
                    if (drawn_active[i]) {
                        draw_needle((float)drawn_angles[i], 0x0000); 
                        drawn_active[i] = 0;
                        need_redraw_center = 1;
                    }
                }
            }

            if (need_redraw_center) {
                drawCircle(RADAR_X, RADAR_Y, 2, 0xFFFF);   
                drawPixel(RADAR_X, RADAR_Y + RADAR_R, 0x07E0); 
            }
            // =========================================================
        }
    }
}