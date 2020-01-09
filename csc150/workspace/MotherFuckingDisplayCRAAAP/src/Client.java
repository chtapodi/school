/** Client for Lab 9
 * 
 * @author 
 * @version 3/2/2017
 */
public class Client
{
    public static void main(String[] args)
    {
	Testing.startTests();

	// Put tests here!
	testBasics();
	
	Testing.finishTests();
    }
    
    
    
    private static void testBasics() {
    	
    	BinarySearchTree tree = new BinarySearchTree<Integer>();
    	
    	for(int i=5;i<10;i+=2) {
    		tree.insert(i);
    	}
    	for(int i=4;i<11;i+=2) {
    		tree.insert(i);
    	}
    	
    	System.out.println(tree.displayTree());

    	
    }

}