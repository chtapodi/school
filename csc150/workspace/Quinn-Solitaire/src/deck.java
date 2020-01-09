import java.util.Random;

public class deck {
	card[] deck;
	Random Random;
	int dealer;
	int deckitterator;
	
	/**
	 * The constructor takes no inputs and fills itself with cards.
	 * @param This constructor takes no params
	 */
	deck() {
		deck = new card[52];
		Random = new Random();
		dealer = 0;
		deckitterator=0;
		for(int i=1;i<5;i++) {
			for(int j=0;j<13;j++) {
				deck[deckitterator] = new card(i,j);
				deckitterator++;
			}
		}
	}
	
	/**
	 * This function evaluates the contents of the deck
	 * @return This returns the contents of the deck in a printable string
	 */
	public String tostring() {
		String alldeck = "The current deck:\n";
		for(int i=0;i<=51;i++){
			alldeck = alldeck + deck[i].toString() + "\n";
		}
		return alldeck;
	}
	
	/**
	 * 
	 * @return This returns the suite value of a new card
	 */
	public int deal() {
		int returnval=deck[dealer].getsuite();
		dealer++;
		return returnval;
		
	}
	/**
	 * Gets the dealer value
	 * @return returns the Int dealer value
	 */
	public int getDealerVal() {
		
		return dealer;
	}
	
	/**
	 * Shuffles the deck
	 */
	public void shuffle() {
		dealer=0;

		for(int i=0;i<=51;i++){
			int randomval = Random.nextInt(51);
			card firstarr = deck[i];
			card secondarr = deck[randomval];
			deck[i] = secondarr;
			deck[randomval] = firstarr;
		}	
	}
	

}
