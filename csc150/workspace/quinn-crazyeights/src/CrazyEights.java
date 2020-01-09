import java.util.Scanner;
public class CrazyEights {
	private final int DEF_DECK_SIZE=104;
	private final int DEF_NUM_SUITES=4;
	private final int DEF_CARD_PER_SUITE=13;
	private final int DEF_INIT_HAND=7;


	private Hand[] player;
	private Scanner scan;
	private int numCards;
	private Deck deck;
	private int turn;
	private String[] playerName;
	private boolean endGame;
	private Card faceCard;
	private String pseudoSuite;
	private Card tmpCard;
	private boolean cont;


	public CrazyEights() {
		scan=new Scanner(System.in);
		deck=new Deck(DEF_DECK_SIZE, DEF_NUM_SUITES, DEF_CARD_PER_SUITE);
		numCards=DEF_DECK_SIZE;

		deck.shuffle();
	}


	public void playGame() {
		System.out.println("input number of players");
		int numPlayers = scan.nextInt();
		endGame=false;
		turn=0;
		pseudoSuite="Nothing, becuase it hasn't been assigned a value yet";
		cont=false;


		if(numPlayers<=1) {
			System.out.println("This is not enough players");
		}

		else if(numPlayers>(numCards/DEF_INIT_HAND)) {
			System.out.println("You do not have a large enough deck");
		}
		else {
			//The actual gameplay
			player = new Hand[numPlayers];
			playerName=new String[numPlayers];

			for(int i=0;i<numPlayers;i++) {
				System.out.println("Please input the name of the next player");
				playerName[i]=scan.next();

			}

			//Makes a hand for each player
			for(int i=0;i<numPlayers;i++) {
				player[i] = new Hand(numCards);
				for(int j=0; j<7;j++) {
					player[i].drawCard(deck.deal());
				}

			}

			//Places the starter card
			faceCard=deck.deal();

			while(endGame==false) {

				//Says who's turn it is
				System.out.println("It is now " + playerName[turn] +"s turn\n");

				//Shows the top card
				System.out.println("The top card is:\n" + faceCard.toString());
				if (faceCard.getValue()==8) {
					System.out.println("But is acting as " + pseudoSuite);
				}

				//Shows their hand
				System.out.println("Their hand:\n");
				player[turn].showHand();
				System.out.println("Input " + (int)(player[turn].handSize()+1) + " to draw\n");



				while (cont==false) {

					//Gets their move
					System.out.println(playerName[turn] + "s move?");

					tmpCard=player[turn].removeCard(scan.nextInt());
					
					if(tmpCard==null) {
						player[turn].drawCard(deck.deal());
						System.out.println("New card is " + player[turn].seekCard(player[turn].handSize()-1).toString());
						cont=true;
					}
					else {

						if(tmpCard.getValue()==8) {
							System.out.println("CRAZY EIGHT\nWhich Suite?\n0) Clubs\n1) Diamonds\n2) Hearts\n3) Spades");
							switch (scan.nextInt()) {

							case 0: pseudoSuite = "Clubs";
							break;
							case 1: pseudoSuite = "Diamonds";
							break;
							case 2: pseudoSuite = "Hearts";
							break;
							case 3: pseudoSuite = "Spades";
							break;
							}
							
							faceCard=tmpCard;
							cont=true;
						}
						else if(tmpCard.getValue()==faceCard.getValue() || tmpCard.getSuite()==faceCard.getSuite() || tmpCard.getValue()==8) {
							faceCard=tmpCard;
							cont=true;
						}
						else {
							System.out.println("This is not a valid move, please try again\n");
						}
					}
				}
				cont=false;


				//If the game is over
				if(player[turn].handSize()==0) {
					System.out.println("GAME OVER");
					endGame=true;
				}
				else if(player[turn].handSize()==1) {
					System.out.println(playerName[turn] + " Only has one card left!");
				}

				//loops
				if(turn==numPlayers-1) {
					turn=0;
				}
				else {
					turn++;
				}
			}
		}
	}
}