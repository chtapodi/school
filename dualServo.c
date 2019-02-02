/* 
 * File:   Servo.c
 * Author: Xavier Quinn
 * Date: 30 January 2019
 * Purpose: To finish the lab
 * Modified:
 */
/*********** COMPILER DIRECTIVES *********/

// #include for textbook library header files
#include "pic24_all.h"
#include "lcd4bit_lib.h"

// #defines for handy constants 
#define LED (_LATA0)  // LED on microstick, RA0 (pin 2)
#define swtch (_RB2) //Defines switch to pin 6

/*********** GLOBAL VARIABLE AND FUNCTION DEFINITIONS *******/

uint16_t pulseWidth;
uint8_t pMinP=117;
uint8_t pMaxP=352;
uint8_t pMinC=203;
uint16_t pMaxC=266;
uint16_t period=3120;
char * toDisplay;






void configTimer2(void) { //Runs the configuration for timer2
    
    T2CON=0x0030;
    //PR2 = msToU16Ticks(250, getTimerPrescale(T2CONbits)) - 1;
    PR2=  period;//0x98BC; //the delay to make PR2 .25seconds 
    _T2IF=0;
    T2CON=0x8030; //turns on T2CON hopefully
   
}
 
void configOC1(void) { //Configuration for PWM
    T2CON=0x0030;
    
    CONFIG_OC1_TO_RP(RB1_RP);
    OC1RS=0x0000;
    OC1R=0x0000;
    OC1CON=0x0006; //sets up the control for PWM on clock2
}



void _ISR _T2Interrupt(void) { //The function that is called on the timer2 interupt

    OC1RS=pulseWidth;
    
    
    _T2IF=0;
    
    
}


uint8_t getServoPos() {
    //get and convert potentiometer val
}

void switchServo(uint8_t val) {
    //switches servo according to val.
}



/********** MAIN PROGRAM ********************************/
int main ( void )  //main function that....
{ 
/* Define local variables */

    
    

/* Call configuration routines */
	configClock();  //Sets the clock to 40MHz using FRC and PLL
    configOC1(); 
    configTimer2();
    configControlLCD();
    initLCD(); //initialize the hitachi lcd

    
    TRISA=0x0000; //sets all A to output
    
    
    _T2IE=1; //Enables interupts
/* Initialize ports and other one-time code */
    CONFIG_RB2_AS_DIG_INPUT();

    

/* Main program loop */
    
	while (1) {	
        switchServo(swtch); //changes switch accordingly
        if (swtch==1) { //continuous
            //set to Continuous pin
            toDisplay="Continuous  ";//sets the correct string to display
        }
        else {
            toDisplay="Positional  "; //sets the correct string to display
        }
        writeLCD(0x80, 0, 1, 1);//resets the courser
        outStringLCD(toDisplay); //displays the string

        pulseWidth=getServoPos(); //calculates the correct value for the pulsewidth
        DELAY_MS(50); //Waits 50ms
        
    }
        
        
}