
public class Hand {
	
	private int nextSpot;
	private Card[] hand;
	
	/**
	 * This contructs a new hand on a per player basis
	 * @param deckSize The size of the deck
	 */
	
	public Hand(int deckSize) {

		nextSpot=0;
		//The largest hand someone can have is theoretically the entire deck -1
		hand = new Card[deckSize-1];
		
	}
	
	/**
	 * This receives a new card and adds it to the hand.
	 * @param newcard is based a card from the shared deck
	 */
	public void drawCard(Card newCard) {
		hand[nextSpot]=newCard;
		//System.out.println(hand[nextSpot].toString() + " is at " + nextSpot);
		nextSpot++;
		
	}
	
	/**
	 *Removes the card at the given index and returns it, as well 
	 * @param The position of the card you want to remove
	 * @return the card you removed
	 */
	public Card removeCard(int cardPos) {
		Card returnCard;
		returnCard=hand[cardPos];
		for(int i = cardPos;hand[i]!=null; i++) {
			hand[ i ] = hand[ i + 1 ];
		}
		return returnCard;
		
		
	}
	/**
	 * Returns the card at the given index but does not remove it
	 * @param cardPos The index of the card to return
	 * @return Returns the card at the index
	 */
	public Card seekCard(int cardPos) {
		return hand[cardPos];
	}
	
	
	/**
	 * Gets the size of the hand at the momment
	 * @return The size of the hand
	 */
	public int handSize() {
		return nextSpot-1;
	}
	
	/**
	 * Shows the current hand, I put this is here because this is someone you want
	 * your hand to do, not only for crazy eights
	 */
	public void showHand() {
		for(int i=0;hand[i]!=null;i++) {
			System.out.print(i + ") " + hand[i] + "\n");
		}
	}

}
