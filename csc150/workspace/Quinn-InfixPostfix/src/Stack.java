// Don't forget the Javadocs!
// Notice that the generic type parameter does NOT implement
// the Token interface.  Make sure you understand why it shouldn't
// (and see the StackTester class for a hint.  Or just ask me!)


/**
* A Stack ADT that holds tokens.
* The first to be inserted into the stack is the last to be removed
* 
*/
public class Stack<T>
{
    private LinkedList can;

    
    /**
     * Default constructor, makes a stack
     * Takes no inputs
     */
    public Stack() {
    	can = new LinkedList<T>(); //The LinkedList is more complex than it needs to be
    							   //But I am reusing one that I already made, 
    							   //Which is good I think. But thats why its 
    							   //Not optimized for a stack
    }
    
    
   /**
    * Checks if the stack is empty
    * @return true if empty, false if not empy
    */
    public boolean isEmpty() {  
    	if(can.getLength()>0) {
    		return false;
    	}
    	else {
    		return true;
    	}
    }

    /**
     * Adds given token to top of stack
     * @param toPush the token to add to the top of the stack
     */
    public void push(T toPush) {
    	can.insertAt(0, toPush);  
    }
    
    /**
     * Removes and returns the top token of the stack
     * @return The top token of the stack, if there is no top token, return null
     */
    public T pop() {
    	return (T)can.removeAt(0);
    } 
  
    /**
     * Looks at the top of the stack, but does not return it.
     * @return The top token of the stack, or null if there isnt one
     */
    public T peek() {
    	return (T)can.getData(0);
    } 
    
    /**
     * Returns the "height" of the stack
     * @return The size of the stack
     */
    public int size() {
    	return can.getLength();
    }
     
    /**
     * Returns the entire stack in string format, read top to bottom. The top value is indicated with a >.
     * @return A string version of the stack
     */
	public String toString() {

		String toReturn = "{>";
		int tester = 0;

		while (can.getData(tester) != null && tester < this.size()) { // While you dont run out of nodes

			toReturn = toReturn + can.getData(tester); // Adds the info

			if (tester < this.size() - 1
					&& can.getData(tester + 1) != null) {
				toReturn += ", ";
			}
			tester++;
		}
		toReturn += "}";
		return toReturn;

	}
    
} 
   

