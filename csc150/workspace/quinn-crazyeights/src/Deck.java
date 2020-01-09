import java.util.Random;



public class Deck {
	final int DEFAULT_DECK_SIZE=52;
	final int DEFAULT_NUM_SUITES=4;
	final int DEFAULT_CARDS_PER_SUITE=13;
	
	Card[] deck;
	Random Random;
	private int dealer;
	private int deckItterator;
	private int deckSize;
	private int numSuites;
	private int cardsPerSuite;
	
	/**
	 * The constructor takes no inputs and fills itself with cards.
	 * @param This constructor takes no params
	 */
	Deck() {
		deck = new Card[DEFAULT_DECK_SIZE];
		deckSize=DEFAULT_DECK_SIZE;
		numSuites=DEFAULT_NUM_SUITES;
		cardsPerSuite=DEFAULT_CARDS_PER_SUITE;
		
		dealer = 0;
		deckItterator=0;
		while(deckItterator<DEFAULT_DECK_SIZE) {
			for(int i=0;i<DEFAULT_NUM_SUITES;i++) {
				for(int j=0;j<DEFAULT_CARDS_PER_SUITE;j++) {
					deck[deckItterator] = new Card(i,j);
					deckItterator++;
				}
			}
		}
	}
	/**
	 * Constructor for the deck
	 * @param newDeckSize The size of the deck
	 * @param newNumSuites The number of suites
	 * @param newCardsPerSuite The number of cards per suite
	 */
	
	Deck(int newDeckSize, int newNumSuites, int newCardsPerSuite) {
		deckSize=newDeckSize;
		deck = new Card[deckSize];
		numSuites=newNumSuites;
		cardsPerSuite=newCardsPerSuite;
		
		
		dealer = 0;
		deckItterator=0;
		while(deckItterator<deckSize) {
			for(int i=0;i<(numSuites);i++) {
				for(int j=0;j<cardsPerSuite;j++) {
					deck[deckItterator] = new Card(i,j);
					deckItterator++;
				}
			}
		}
	}
	
	
	
	/**
	 * This function evaluates the contents of the deck
	 * @return This returns the contents of the deck in a printable string
	 */
	public String tostring() {
		String alldeck = "The current deck:\n";
		for(int i=0;i<deckSize;i++){
			alldeck = alldeck + deck[i].toString() + "\n";
		}
		return alldeck;
	}
	
	/**
	 * 
	 * @return This returns a new card
	 */
	public Card deal() {
		Card returnval=deck[dealer];
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
	 * This is not used, but I feel like it would be useful if I have to use this again
	 * @return The number of suites
	 */
	public int numSuites() {
		return numSuites;
		
	}
	
	/**
	 * Shuffles the deck
	 */
	public void shuffle() {
		dealer=0;
		Random = new Random();
		for(int i=0;i<deckSize;i++){
			int randomval = Random.nextInt(deckSize);
			Card firstarr = deck[i];
			Card secondarr = deck[randomval];
			deck[i] = secondarr;
			deck[randomval] = firstarr;
		}	
	}
	

}
