import CSLib.DrawingBox;
import CSLib.OutputBox;
import java.awt.Rectangle;

public class RectangleTester {

	/**
	 * Xavier Quinn
	 * The purpose of this class is to create a rectangle using java.awt.Rectangle and perform various actions with its library.
	 * Lab 2
	 * 01.12.17
	 */
	
	public static void main(String[] args) {
		//This creates several DrawingBox s
		
		DrawingBox myBoard, secondBoard;
		myBoard = new DrawingBox();
		myBoard.setVisible(true);
		secondBoard = new DrawingBox();
		secondBoard.setVisible(true);
		//myBoard.drawRect(320,230,120,180);
		
		
		//This makes several new rectangles
		Rectangle myRect,Rect1,Rect2,union;
		myRect = new Rectangle(320,230,120,180);
		Rect1 = new Rectangle(50,180,120,180);
		Rect2 = new Rectangle(100,240,150,200);
		
		
		//This makes an OutputBox
		OutputBox boxOut;
		boxOut = new OutputBox();	
		
		
		//This messes with the rectangles on myBoard
		myBoard.drawRect(myRect);
	
		myRect.grow(20, 20);
		myBoard.drawRect(myRect);
		
		myRect.translate(280, -50);
		myBoard.drawRect(myRect);
		
		myRect.setLocation(75, 250);
		myBoard.drawRect(myRect);
		
		myRect.setLocation(75, 50);
		myRect.setSize(200, 150);
		myBoard.drawRect(myRect);
		
		
		//This section performs tests on rectangles on secondBoard, such as checking for intersection and seeing if one
		//contains the other two.
		
		secondBoard.drawRect(Rect1);
		secondBoard.drawRect(Rect2);
		
		if (Rect1.intersects(Rect2)) {
			boxOut.println("The two boxes intersect intersect");
		}
		else {
			boxOut.println("Somehow the boxes don't intersect");
		}
		
		union = new Rectangle(Rect1.union(Rect2));
		secondBoard.drawRect(union);
		
		if (union.contains(Rect2) && union.contains(Rect1)) {
			boxOut.println("The union box contains both of the other rectangles");
		}
		else {
			boxOut.println("Somehow the box doesn't contain both rectangles");
		}
	}
}
