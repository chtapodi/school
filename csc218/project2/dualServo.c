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
uint8_t pMinP=120;
uint16_t pMaxP=352;
uint8_t pMinC=120;
uint16_t pMaxC=266;
uint16_t period=3125;
char * toDisplay;

//to allow dynamic association between servo positions.
uint8_t minVal;
uint8_t maxVal;






void configTimer2(void) { //Runs the configuration for timer2
    
    T2CON=0x0030; //turn t2 off
    //PR2 = msToU16Ticks(250, getTimerPrescale(T2CONbits)) - 1;
    PR2=  period;//0x98BC; //the delay to make PR2 .25seconds 
    _T2IF=0;
    T2CON=0x8030; //turns on T2CON hopefully
   
}
 
void updateServo(void ) {

}

void configOC() { //Configuration for PWM
    T2CON=0x0030;
    
    

    CONFIG_OC1_TO_RP(RB11_RP);

    CONFIG_OC2_TO_RP(RB12_RP);

    OC1RS=0x0000;
    OC1R=0x0000;
    OC1CON=0x0006; //sets up the control for PWM on clock1
    
    OC2RS=0x0000;
    OC2R=0x0000;
    OC2CON=0x0006; //sets up the control for PWM on clock2
}



void _ISR _T2Interrupt(void) { //The function that is called on the timer2 interupt

    if (swtch) { //continous
    OC1RS=pulseWidth;
    }
    else {
        OC2RS=pulseWidth;
    }
    
    
    _T2IF=0; //resets the timer
    
    
}


uint16_t getServoPos(void) {
    //get and convert potentiometer val
    //float posVal = convertADC1(); //max val of 1024
    float posVal=convertADC1()/1025.0;
    float diff;
    if(swtch) {
        diff=146;
    }
    else {
        diff=230;
    }
    

    
    return posVal*diff +120;
}





/********** MAIN PROGRAM ********************************/
int main ( void )  //main function that....
{ 
/* Define local variables */

    
    

/* Call configuration routines */
	configClock();  //Sets the clock to 40MHz using FRC and PLL
    configOC(); 
    configTimer2();
    configControlLCD();
    initLCD(); //initialize the hitachi lcd
    
    
    TRISA=0x0000; //sets all A to output MUST RE_EVALUATE
    CONFIG_RB2_AS_DIG_INPUT();
    CONFIG_RA0_AS_ANALOG(); //sets up potentiometer

    configADC1_ManualCH0(RA0_AN, 1, 0); //sets range

    
    
    
    
    _T2IE=1; //Enables interupts
/* Initialize ports and other one-time code */
    CONFIG_RB2_AS_DIG_INPUT();

    

/* Main program loop */
    
	while (1) {	

        if (swtch) { //continuous
            //set to Continuous pin
            toDisplay="Continuous  ";//sets the correct string to display

        }
        else {
            toDisplay="Positional  "; //sets the correct string to display

        }
        char array[50];
        writeLCD(0x80, 0, 1, 1);//resets the courser
        
        char str[100];
        sprintf(str, "%d", getServoPos());

        outStringLCD(str); //displays the string

        pulseWidth=getServoPos(); //calculates the correct value for the pulsewidth
        
    }
        
        
}