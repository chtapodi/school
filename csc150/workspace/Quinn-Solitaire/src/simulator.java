public class simulator {
	int topcard;
	int[] hand;
	deck deck;
	final int DECKSIZE=52;

	/**
	 * Default constructor takes no inputs
	 */
	public simulator() {
		topcard=0;
		hand=new int[DECKSIZE]; // 52 cards in a deck
		deck=new deck();
	}

	/**
	 * This method goes through and plays the game according to the rules
	 * @return Returns whether the game was a success or not
	 */
	public boolean playgame() {
		boolean outcome = false;
		deck.shuffle(); //Shuffles the cards

		for (int i = 0; i < 4; i++) { // Initiates a game by dealing 4 cards
			hand[i] = deck.deal();
		}
		
		topcard=3; //Tells the simulator how many cards there are to start

		while (deck.getDealerVal() <= 51) {
			if (topcard >= 3) {
				check();
				topcard++;
				
				if (topcard >= 3) {
					hand[topcard] = deck.deal();
				}
			}
			else if(topcard<0) {//win condition
				outcome = true;
				return outcome;
			}
			else {
				refill();
			}
		}
		return outcome; //Loss condition

	}
	
	/**
	 * This method removes the selected range of cards and resorts them so there are no spaces
	 * @param second Positive range value for what is to be deleted
	 * @param first Positive range value for what is to be deleted
	 */
	private void remove(int second, int first) {
		if (hand[second + 1] != 0) {
			int transfer = hand[second + 1];
			hand[second+1] = 0;
			hand[first] = transfer;
			hand[second] = 0;
			topcard = second-1;
		} 
		
		else {
			for (int i = second; i >= first; i--) {
				hand[i] = 0;
				topcard--;
			}
			topcard--;
		}
	}
	
	/**
	 * This method checks if any cards can be removed, and if so calls remove()
	 */
	
	private void check() {
		while (topcard>=3 && hand[topcard] == hand[topcard - 3]) {
			if (hand[topcard] == hand[topcard - 2]
					&& hand[topcard] == hand[topcard - 1]) {
				remove(topcard, (topcard - 3));;
			} else {
				remove((topcard - 1), (topcard - 2));
				hand[topcard + 3] = hand[topcard + 1];
			}
		}
	}
	
	/**
	 * This refills the current hand; for use when there are less than four cards
	 */
	private void refill() {
		while (topcard < 3 && deck.getDealerVal()<=51) { //Adds cards if there are not enough
			hand[topcard] = deck.deal();
			topcard++;
		}
	}

}
