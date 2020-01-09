/**
 * Linked List is a collection of data nodes. All methods here relate to how one
 * can manipulate those nodes.
 * 
 * @author Xavier
 * @version 02.09.17
 * 
 *          * I affirm that I have carried out the attached academic endeavors
 *          with full academic honesty, in accordance with the Union College
 *          Honor Code and the course syllabus.
 */
public class LinkedList<T> {
	private int length; // number of nodes
	private ListNode firstNode; // pointer to first node

	public LinkedList() {
		length = 0;
		firstNode = null;
	}

	/**
	 * Inserts at specified location
	 * 
	 * @param The index at which to insert, starting at 0
	 * @param The thing to insert
	 */
	public void insertAt(int place, T toInsert) {
		ListNode<T> newNode = new ListNode<T>(toInsert);

		if (getLength() == 0) {
			firstNode = newNode; // If theres nothing in the list, insert at
									// start
		}

		else {
			ListNode<T> n;
			n = firstNode;
			int index = 0;
			while (index != (this.getLength() - 1) && index < (place - 1)) { 
				n = n.next;
				index++;
			}

			if (index == this.getLength()) { // adds to end if place is above
												// end
				n.next = newNode;
			} else if (place <= 0) { // if place is 0 or less, insert at start
				newNode.next = firstNode; //
				firstNode = newNode;
			}

			else { // n.next is 'place', adds before place.
				newNode.next = n.next;
				n.next = newNode;
			}
		}
		length++;
	}

	/**
	 * Removes the data from the given location and returns it
	 * 
	 * @param The
	 *            location to remove, starting at 0
	 * @return The removed information
	 */

	public T removeAt(int place) {
		if (place < 0 || place >= this.getLength())
			return null;

		ListNode<T> n;
		n = firstNode;
		T toReturn;

		if (place == 0) { // If its the first node
			toReturn = n.data;
			firstNode = n.next;
			length--;
			return toReturn;
		}

		int index = 0;
		while (index != (this.getLength() - 1) && index < (place - 1)) { 
			n = n.next;
			index++;
		}

		toReturn = (T) (n.next.data); // returns the data at the place
		n.next = n.next.next;

		length--;
		return toReturn;
	}

	/**
	 * Turn entire chain into a string
	 * 
	 * @return return linked list as printable string of format (string, string,
	 *         string)
	 */
	public String toString() {
		String toReturn = "(";
		ListNode<T> n;
		n = firstNode;
		while (n != null) {
			toReturn = toReturn + n.toString();
			n = n.next;
			if (n != null) {
				toReturn = toReturn + ", ";
			}
		}
		toReturn = toReturn + ")";
		return toReturn;
	}

	/**
	 * getter for number of nodes in the linked list
	 * 
	 * @return length of LL
	 */
	public int getLength() {
		return length;
	}

	/**
	 * Gets the information from the specified location
	 * 
	 * @param The
	 *            index of what to get, starting at 0
	 * @return the information
	 */
	public T getData(int place) {
		if (place < 0 || place >= this.getLength())
			return null;

		ListNode<T> n;
		n = firstNode;
		T toReturn;
		if (place == 0) {
			toReturn = n.data;
			return toReturn;
		}

		int index = 0;
		while (index != (this.getLength() - 1) && index < (place - 1)) {
			n = n.next;
			index++;
		}
		toReturn = (T) (n.next.data);
		return toReturn;
	}

	/**
	 * Clears the linked list of everything
	 */
	public void clear() {
		firstNode = null;
		length = 0;
	}

}