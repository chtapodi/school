/**
 * Sorts items using a PQ.
 * 
 * @author Chris Fernandes
 * @version 3/3/12
 */
public class Sorter
{
	/**
	 * Sorts the elements using a priority queue.  Just make a PQ out of
	 * the given array and then remove all of the items (since they'll
	 * come out in sorted order)!  This method is non-destructive.
	 * The original array is NOT changed.
	 *
	 * @param array an unsorted array of elements
	 * @return a (different) sorted array of elements
	 */
	public static int[] priorityQueueSort(int[] array)
	{  
		int[] answer = new int[array.length];    // array to store the result
		PriorityQueue PQ = new PriorityQueue(array); 

		for (int i = array.length - 1; i >= 0; i--) {
			answer[i] = PQ.remove();
		}
		return answer;
	}

}