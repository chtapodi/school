/**
 * The Heap ADT.  This is a max heap. 
 * 
 *  *          * I affirm that I have carried out the attached academic endeavors
 *          with full academic honesty, in accordance with the Union College
 *          Honor Code and the course syllabus.
 * 
 * @author (Xavier) 
 * @version (march 09 2017)
 */
public class Heap
{
	private int[] itemArray;    //binary tree of ints in array form (element 0 not used) 
	private int nodes;          //number of nodes in the tree

	/**
	 * Builds a heap from an array of ints.
	 *
	 * @param items 
	 *            an array of ints(starting at index 0), which will be 
	 *            interpreted as a binary tree.
	 */
	public Heap(int[] items)
	{
		itemArray = new int[items.length + 1];
		nodes = items.length;

		for (int i = 0; i < nodes; i++) {
			itemArray[i + 1] = items[i];
		}

		buildAHeap();
	}

	/**
	 * @return number of nodes in the heap.
	 */
	public int size()
	{
		return nodes;
	}

	/**
	 * Constructs a heap from the given binary tree (given as an array).  
	 * Heapifies each internal node in reverse level-by-level order.
	 */
	public void buildAHeap()
	{
		for (int i = nodes/2; i >= 1; i--) {
			heapify(i);
		}

	}

	/** string representation of a heap that looks (a little) like a tree
	 * @return string with one int on 1st line, two ints on 2nd line, four ints on 3rd line, etc.
	 */
	public String toString()
	{
		String result = "";
		int lastNodeOnLevel = 1;

		for (int i = 1; i < nodes; i++)
		{
			result += itemArray[i];
			if (i == lastNodeOnLevel) {
				result += "\n";
				lastNodeOnLevel = lastNodeOnLevel * 2 + 1;
			} else {
				result += " ";
			}
		}
		result += itemArray[nodes];

		return result;
	}

	/**
	 * Turns a subtree into a heap, assuming that only the root of that subtree 
	 * violates the heap property.
	 *
	 * @param startingNode 
	 * 			the index of the node to start with.  This node 
	 * 			is the root of a subtree which must be turned into a heap.
	 */
	public void heapify(int start)
	{
		int left=start*2;
		int right=start*2+1;
		int select=start;
		
		if(left<=nodes && itemArray[select]<itemArray[left]) {
			select=left;
		}
		
		if(right<nodes && itemArray[select]<itemArray[right]) {
			select=right;
		}
		
		if(select!=start) {
			replace(start, select);
			heapify(select);
		}
		
		
	}

	
	
	/**
	 * switches the values of the two given nodes
	 * @param parent Node one
	 * @param child Node two
	 */
	private void replace(int parent, int child) {
		int tmp=itemArray[parent];
		itemArray[parent]=itemArray[child];
		itemArray[child]=tmp;
	}

	
	/**
	 * Removes the root from the heap, returning it.  The resulting array is 
	 * then turned back into a heap. 
	 *
	 * @return the root value.
	 */
	public int deleteRoot()
	{
		
		if(nodes<=0) {
			return 0;
		}
		else {
			//System.out.println("ToString Before\n " +this.toString() + "\n");
			int toReturn = itemArray[1];
			//System.out.println("endNode " +itemArray[nodes] + "\n");
			itemArray[1]=itemArray[nodes];
			//System.out.println("ToString middle\n " +this.toString() + "\n");
			nodes--;		
			heapify(1);
			System.out.println(nodes);
		
			return toReturn;
		}
	}

}