package second;
import CSLib.DrawingBox;
import java.awt.Rectangle;

/**
 * Creates and manipulate three dimensional blocks with xyz dimentions
 * and xy position
 * @author Xavier
 *
 */



public class Block {
	
	//constants
	final Rectangle DEFAULT_RECT = new Rectangle(100,100,50,50);
	final int DEFAULT_DEPTH = 50;
	
	private Rectangle rect;
	private int depth;
	
	/**
	 * Constructs the default block with default size and postion
	 * @param takes no inputs
	 */
	public Block(){
		rect=DEFAULT_RECT;
		depth=DEFAULT_DEPTH;
		
	}
	
	
	/**
	 * Constructs a block with default sizes and given coordinates
	 * @param xpos The x position of the block, absolute value of input
	 * @param ypos The y position of the block, absolute value of input
	 */
	public Block(int setxpos, int setypos) {
		depth=DEFAULT_DEPTH;
		rect = DEFAULT_RECT;
		rect.setLocation(Math.abs(setxpos), Math.abs(setypos));

	}
	
	
	
	//Getter methods 
	
	
	/**
	 * @return gets the x size of the block
	 */
	public int getXSize(){
		return (int)(rect.getWidth());
	}
	
	/**
	 * @return gets the y size of the block
	 */
	public int getYSize(){
		return (int)(rect.getHeight());
	}
	
	/**
	 * @return gets the z size of the block
	 */
	public int getZSize(){
		return depth;
	}
	
	/**
	 * @return gets the x position of the block
	 */
	public int getXPos(){
		return (int)(rect.getX());
	}
	
	
	/**
	 * @return gets the y position of the block
	 */
	public int getYPos(){
		return (int)(rect.getY());
	}
	
	
	//Setter methods
	
	/**
	 * Changes the coordinates of the block
	 * @param newx is the new x position, uses absolute value
	 * @param newy is the new y position, uses absolute value
	 */
	public void setPosition(int newxpos, int newypos) {
		rect.setLocation(Math.abs(newxpos), Math.abs(newypos));
	}
	
	
	/**
	 * Changes the dimentions of the block
	 * @param newxsize the new x size, uses absolute values
	 * @param newysize the new y size, uses absolute values
	 * @param newzsize the new z size, uses absolute values
	 */
	public void setDimenstions(int newxsize, int newysize, int newzsize) {
		rect.setSize(Math.abs(newxsize), Math.abs(newysize));
		depth=Math.abs(newzsize);
	}
	
	
	/**
	 * Displays the block on the given drawing box
	 * @param box is a drawingbox to display the box on.
	 */
	public void display(DrawingBox box) {
		box.drawRect(rect);
		int xpos=(int)(rect.getX());
		int ypos=(int)(rect.getY());
		for(int i=0;i<depth;i++){
			rect.setLocation(xpos-(2*i), ypos-(2*i));
			box.drawRect(rect);
		}
	}
}
