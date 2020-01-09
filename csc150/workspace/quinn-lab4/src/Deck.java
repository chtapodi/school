import java.util.Vector;
import java.util.Random;

/**
 * Simulates a deck of playing cards.  This deck has a few more
 * methods than the one you built for Project 2.
 */
public class Deck {
    private final int NUMBER_OF_CARDS=52;
    private final int NUMBER_OF_SUITS=4;
    private final int CARDS_IN_SUIT=13;
    
    private Vector<Card> theCards;
    private boolean shuffled;
    
    /**
     * Makes a new ordered deck of playing cards
     */
    public Deck() {
        theCards = new Vector<Card>(NUMBER_OF_CARDS);
        shuffled=false;
        for(int i=0;i<(NUMBER_OF_SUITS);i++) {
			for(int j=1;j<(CARDS_IN_SUIT+1);j++) {
				theCards.add(new Card(j,i));

			}
		}
        
    }
    
    /**
     * Deals out next card in deck; returns null if no cards left
     *  
     * @return next card in deck or null if deck empty
     */
    public Card deal() {
    	Random randval = new Random();
    	theCards.trimToSize();
    	
    	
    	if(isEmpty()) {
    		System.out.println("is empty");
    		return null;
    	}
    	if(shuffled) {
    		System.out.println("size: " + size());
    		return theCards.remove(randval.nextInt(size()));
    		
    	}
    	else {
    		;
    		return theCards.remove(0);
    		
    	}
        
    }
    
    /** Tells if deck has any cards left in it
     * 
     * @return true if Deck empty; else false
     */
    public boolean isEmpty() {

    	return theCards.isEmpty();
    }
    
    /**
     * Shuffles the cards
     */
    public void shuffle() {
    	shuffled=true;
        
    }
    
    /** Returns number of undealt cards left in the deck
     * 
     * @return number of undealt cards in the deck
     */
    public int size()
    {
    	theCards.trimToSize();

    	return theCards.capacity();
    }
    
    /**
     * Reset the deck by gathering up all dealt cards. 
     * Postcondition: Deck contains all cards and is shuffled
     */
    public void gather() {
    	theCards.clear();
    	for(int i=0;i<(NUMBER_OF_SUITS);i++) {
			for(int j=1;j<(CARDS_IN_SUIT+1);j++) {
				theCards.add(new Card(j,i));
			}
		}
    	shuffle();

    }
    
    /** 
     *  DEBUGGING METHOD: prints out stats of deck, that is, the
     *  <i>undealt</i> cards.  Prints the remaining number of cards of
     *  each suit and of each value.
     */
    public void printStats()
    {
        int Hcount=0;
        int Dcount=0;
        int Scount=0;
        int Ccount=0;
        int[] values = new int[CARDS_IN_SUIT];
        int size=theCards.size();
        for (int i=0; i<size; i++)
        {
            int val = theCards.elementAt(i).getValue();
            String suit = theCards.elementAt(i).getSuit();
            if (suit.equals("clubs"))
                Ccount++;
            else if (suit.equals("diamonds"))
            	Dcount++;
            else if (suit.equals("spades"))
            	Scount++;
            else if (suit.equals("hearts"))
                Hcount++;
	    // deck values run from 1-13 so need to subtract 1
            values[val-1]++;  
        }
        System.out.println("***PRINTING DECK STATS***");
        System.out.println("# clubs: " + Ccount);
        System.out.println("# diamonds: " + Dcount);
        System.out.println("# hearts: " + Hcount);
        System.out.println("# spades: " + Scount);
        
        System.out.print("Card:\t");
        for (int j=0; j<values.length; j++) {
        	System.out.print(Card.values[j]+"\t");
        }
        System.out.println();
        System.out.print("Qty:\t");
        for (int j=0; j<values.length; j++) {
        	System.out.print(values[j] + "\t");
        }
        System.out.println("\n");
    }
}