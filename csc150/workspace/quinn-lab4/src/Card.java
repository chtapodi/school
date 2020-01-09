/**
 * Simulates a playing card with value and suit
 */
public class Card
{
    /** All suits in play **/
    public static final String[] suits={"spades","clubs",
					"hearts","diamonds"};
    
    /** All values in play **/
    public static final String[] values={"Ace","2","3","4","5","6",
					 "7","8","9","10","Jack",
					 "Queen","King"};
    
    private int value;
    private String suit;

    /**
     * Creates a playing card.
     * 
     * @param new_value int between 1-13 (11=J, 12=Q, 13=K, 1=A)
     * @param new_suit suit as a string, i.e. "spades", "clubs",
     * "hearts", or "diamonds"
     */
    public Card(int new_value, String new_suit)
    {
        value = new_value;
        suit = new_suit.toLowerCase();
    }
    
    /** Creates a playing card.
     * @param new_value int between 1-13 (11=J, 12=Q, 13=K, 1=A)
     * @param new_suit int between 0-3 (0=spades, 1=clubs, 2=hearts,
     * 3=diamonds)
     */
    public Card(int new_value, int new_suit)
    {
        value = new_value;
        suit = suits[new_suit];
    }

    /** Getter for value
     * 
     * @return 1 for Ace, 11 for Jack, 12 for Queen, 13 for King, rest
     * are as numbered
     */
    public int getValue()
    {
        return value;
    }
    
    /** Getter for suit
     * 
     * @return "spades", "diamonds", "hearts", or "clubs"
     */
    public String getSuit()
    {
        return suit;
    }
    
    /**  Gives full name of card
     * @return full name of card, such as "Jack of clubs"
     */
    public String toString()
    {
        String answer = values[getValue()-1];
        answer+=" of " + getSuit();
        return answer;
    }
}