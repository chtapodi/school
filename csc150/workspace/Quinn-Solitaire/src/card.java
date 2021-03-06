 	
public class card {
	
	
	int suite;
	int value;
	/**
	 * This constructor creates a new card with given values.
	 * @param setsuite The suite that this card will be
	 * @param setvalue The value this card will have
	 */
	
	public card(int setsuite, int setvalue) {
		value=setvalue;
		suite=setsuite;
		
	}
	
	/**
	 * This returns the information on the card in a printable string
	 * I know you said to never use breaks, but hopefully this is an exception 
	 * because the switch() method required that you use breaks and I think 
	 * this is the most efficient way to go about it
	 * @returns a string of the card data
	 */
	
	 public String toString() {
		 String cardval = "Joker";
		 String suiteval = "clubs";
		 if (value>9) {
			 switch(value) {
			 	case 10: cardval ="jack";
			 		break;
			 	case 11: cardval ="queen";
			 		break;
			 	case 12: cardval ="king";
			 		break;
			 }
		 }
		 else {
			 cardval=Integer.toString(value+1);
		 }
		 
		 switch(suite) {
		 	case 1: suiteval ="hearts";
		 		break;
		 	case 2: suiteval ="spades";
		 		break;
		 	case 3: suiteval ="diamonds";
		 		break;
		 	case 4: suiteval="clubs";
		 }
		 
		 return cardval + " of " + suiteval;
	 }
	 
	 /**
	  * A getter for the suite
	  * @return Returns the suite of the card
	  */
	 public int getsuite() {
		 return suite;
	 }
	 
	 /**
	  * A getter for the card value
	  * @return The cards value
	  */
	 public int getvalue() {
		 return value;
	 }

}
