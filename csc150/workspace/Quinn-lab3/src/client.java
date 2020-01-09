import second.Block;
import CSLib.DrawingBox;


/**
 * This is to test out the new blocks class 
 * @author xavier
 *
 *I affirm that I have carried out the attached 
 *academic endeavors with full academic honesty, in
 *accordance with the Union College Honor Code and
 *the course syllabus.
 *
 */
public class client {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		//Constructing my board and my block
		DrawingBox myBoard = new DrawingBox();
		DrawingBox myBoard2 = new DrawingBox();
		Block myblock = new Block();
		Block myblock2 = new Block();
		Block[] blockarray = new Block[5];
		

		
		
		
		//makes it so you can see the board
		myBoard.setVisible(true);
		
		//sets potisition of second block
		myblock2.setPosition(100,250);
		
		//draws both boxes
		myblock2.display(myBoard);
		myblock.display(myBoard);
		

		//Moves first box and shows it
		myblock.setPosition(300, 175);
		myblock.display(myBoard);
		
		//moves the second block and shows it
		myblock2.setPosition(200, 400);
		myblock2.display(myBoard);
		
		
		myblock.setPosition(700, 200);
		myblock.setDimenstions(2*myblock.getXSize(), myblock.getYSize(), 2*myblock.getZSize());
		myblock.display(myBoard);
		
		//makes an array of 5 blocks and displays them
		for(int i=0;i<5;i++) {
			blockarray[i]=new Block(100, 100*(i+1));
			blockarray[i].display(myBoard2);
		}
		
		
		
		

	}

}
