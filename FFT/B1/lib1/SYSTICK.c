#include "SYSTICK.h"

volatile uint32_t systick_ms = 0;



void SysTick_Init(void)
{

    SysTick->LOAD = 100000 - 1;   
    SysTick->VAL  = 0;

    SysTick->CTRL = (1<<2) |     // Clock = AHB
                    (1<<1) |     // Enable interrupt
                    (1<<0);      // Enable SysTick
}



void SysTick_Handler(void)
{
    systick_ms++;
}



uint32_t millis(void)
{
    return systick_ms;
}



void Delay_ms(uint32_t ms)
{
    uint32_t start = millis();

    while(millis() - start < ms);
}
