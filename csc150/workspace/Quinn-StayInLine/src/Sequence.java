/**
 * Sequence is an abstract data type that acts as a disk storing strings. You
 * can advance position and add strings to before and after the current
 * position.
 * 
 * Current will never be less than 0. If it does not exist, it will be a value
 * greater than the number of items in the sequence
 * 
 * There will never be empty spaces between elements in the Sequence, and the
 * spaces after the elements will be null.
 * 
 * @author xavier
 */
public class Sequence {

	private int current;
	private int capacity;
	private LinkedList<String> seq;

	public Sequence() {
		seq = new LinkedList<String>();
		capacity = 10;
		current = this.size() + 1;
	}

	/**
	 * Creates a new sequence.
	 * 
	 * @param initialCapacity
	 *            the initial capacity of the sequence.
	 */
	public Sequence(int initialCapacity) {
		seq = new LinkedList<String>();
		capacity = initialCapacity;
		current = this.size() + 1;
	}

	/**
	 * Adds a string to the sequence in the location before the current element.
	 * If the sequence has no current element, the string is added to the
	 * beginning of the sequence.
	 * 
	 * The added element becomes the current element.
	 * 
	 * If the sequences's capacity has been reached, the sequence will expand to
	 * twice its current capacity plus 1.
	 * 
	 * @param value
	 *            the string to add.
	 */
	public void addBefore(String value) {

		this.sizeCheck();

		if (current > this.size()) {
			seq.insertAt(0, value);
			current = 0;
		} else {
			seq.insertAt(current, value);
		}

	}

	/**
	 * Adds a string to the sequence in the location after the current element.
	 * If the sequence has no current element, the string is added to the end of
	 * the sequence.
	 * 
	 * The added element becomes the current element.
	 * 
	 * If the sequences's capacity has been reached, the sequence will expand to
	 * twice its current capacity plus 1.
	 * 
	 * @param value
	 *            the string to add.
	 */
	public void addAfter(String value) {
		this.sizeCheck();
		if (current > this.size()) {
			seq.insertAt(this.size() + 1, value);
			current = 0;
		} else {

			seq.insertAt(current + 1, value);
			current++;
		}

	}

	/**
	 * Returns true if and only if the sequence has a current element.
	 * 
	 * @return true if and only if the sequence has a current element.
	 */
	public boolean isCurrent() {
		if (current != -1 && seq.getData(current) != null) {
			return true;
		}
		return false;
	}

	/**
	 * @return the capacity of the sequence.
	 */
	public int getCapacity() {
		return capacity;
	}

	/**
	 * @return the element at the current location in the sequence, or null if
	 *         there is no current element.
	 */
	public String getCurrent() {
		if (this.isCurrent()) {
			return seq.getData(current);
		} else {
			return null;
		}
	}

	/**
	 * Increase the sequence's capacity to be at least minCapacity. Does nothing
	 * if current capacity is already >= minCapacity.
	 * 
	 * @param minCapacity
	 *            the minimum capacity that the sequence should now have.
	 */
	public void ensureCapacity(int minCapacity) {
		if (this.getCapacity() < minCapacity) {
			scaleTo(minCapacity);
		}

	}

	/**
	 * Places the contents of another sequence at the end of this sequence.
	 * 
	 * If adding all elements of the other sequence would exceed the capacity of
	 * this sequence, the capacity is changed to make room for all of the
	 * elements to be added.
	 * 
	 * Postcondition: NO SIDE EFFECTS! the other sequence should be left
	 * unchanged. The current element of both sequences should remain where they
	 * are. (When this method ends, the current element should refer to the same
	 * element that it did at the time this method started.)
	 * 
	 * @param another
	 *            the sequence whose contents should be added.
	 */
	public void addAll(Sequence another) {
		Sequence tmpSeq = another.clone();
		int maxSize = (another.size() + this.size());

		if (current > this.size()) {
			current = (this.size() + another.size() + 1);
		}

		// If too small
		if (this.getCapacity() < (another.size() + this.size())) {
			scaleTo((another.size() + this.size()));
		}

		tmpSeq.start();

		for (int i = this.size(); i < maxSize; i++) {

			seq.insertAt(i, tmpSeq.getCurrent());

			tmpSeq.advance();
		}
	}

	/**
	 * Move forward in the sequence so that the current element is now the next
	 * element in the sequence.
	 * 
	 * If the current element was already the end of the sequence, then
	 * advancing causes there to be no current element.
	 * 
	 * If there is no current element to begin with, do nothing.
	 */
	public void advance() {
		if (current + 1 == this.size() || current == -1) { // So I am not sure
															// if by the end of
															// the sequence you
															// mean
			// end of the values or end of the capacity, or we decide.
			current = -1; // So I have decided that as part of my invariant
							// current can never be on a null

		} else {
			current++;
		}

	}

	/**
	 * Make a copy of this sequence. Subsequence changes to the copy do not
	 * affect the current sequence, and vice versa.
	 * 
	 * Postcondition: NO SIDE EFFECTS! This sequence's current element should
	 * remain unchanged. The clone's current element will correspond to the same
	 * place as in the original.
	 * 
	 * @return the copy of this sequence.
	 */
	public Sequence clone() /* Sequence */
	{
		Sequence newSeq = new Sequence(this.getCapacity());

		for (int i = 0; i < this.size(); i++) {
			newSeq.addAfter(seq.getData(i));
		}

		return newSeq;
	}

	/**
	 * Remove the current element from this sequence. The following element, if
	 * there was one, becomes the current element. If there was no following
	 * element (current was at the end of the sequence), the sequence now has no
	 * current element.
	 * 
	 * If there is no current element, does nothing.
	 */
	public void removeCurrent() {
		seq.removeAt(current);

	}

	/**
	 * @return the number of elements stored in the sequence.
	 */
	public int size() {

		return seq.getLength();
	}

	/**
	 * Sets the current element to the start of the sequence. If the sequence is
	 * empty, the sequence has no current element.
	 */
	public void start() {
		if (this.isEmpty()) {
			current = -1;
		} else {
			current = 0;
		}
	}

	/**
	 * Reduce the current capacity to its actual size, so that it has capacity
	 * to store only the elements currently stored.
	 */
	public void trimToSize() {
		scaleTo(this.size());
	}

	/**
	 * Produce a string representation of this sequence. The current location is
	 * indicated by a >. For example, a sequence with "A" followed by "B", where
	 * "B" is the current element, and the capacity is 5, would print as:
	 * 
	 * {A, >B} (capacity = 5)
	 * 
	 * The string you create should be formatted like the above example, with a
	 * comma following each element, no comma following the last element, and
	 * all on a single line. An empty sequence should give back "{}" followed by
	 * its capacity.
	 * 
	 * @return a string representation of this sequence.
	 */
	public String toString() {

		String toReturn = "{";
		int tester = 0;

		while (seq.getData(tester) != null && tester < this.getCapacity()) { // While you dont run out of nodes

			if (tester == current) { // Add > for the current string
				toReturn = toReturn + ">";
			}

			toReturn = toReturn + seq.getData(tester); // Adds the info

			if (tester < this.getCapacity() - 1
					&& seq.getData(tester + 1) != null) {
				toReturn += ", ";
			}
			tester++;
		}
		toReturn += "} (capacity = " + this.getCapacity() + ")";
		return toReturn;

	}

	/**
	 * Checks whether another sequence is equal to this one. To be considered
	 * equal, the other sequence must have the same size as this sequence, have
	 * the same elements, in the same order, and with the same element marked
	 * current. The capacity can differ.
	 * 
	 * Postcondition: NO SIDE EFFECTS! this sequence and the other sequence
	 * should remain unchanged, including the current element.
	 * 
	 * @param other
	 *            the other Sequence with which to compare
	 * @return true if the other sequence is equal to this one.
	 */
	public boolean equals(Sequence other) {
		if (this.size() != other.size()) {
			return false;
		}

		Sequence cloneOne = this.clone();
		Sequence cloneTwo = other.clone();
		cloneOne.start();
		cloneTwo.start();

		for (int i = 0; i < this.size(); i++) {
			if (cloneOne.getCurrent() == null
					|| cloneTwo.getCurrent() == null
					|| cloneOne.getCurrent().compareTo(cloneTwo.getCurrent()) != 0) {
				return false;
			}
			cloneOne.advance();
			cloneTwo.advance();
		}

		if (this.getCurrent() != null && other.getCurrent() != null
				&& this.getCurrent().compareTo(other.getCurrent()) != 0) {
			return false;
		}

		return true;
	}

	/**
	 * 
	 * @return true if Sequence empty, else false
	 */
	public boolean isEmpty() {
		if (this.size() == 0) {
			return true;
		}
		return false;
	}

	/**
	 * empty the sequence. There should be no current element.
	 */
	public void clear() {
		seq.clear();
		current = -1;
	}

	/**
	 * If adding to the sequence will overfill it, make it 2x the size plus one
	 */
	private void sizeCheck() {
		if (this.getCapacity() < this.size() + 1) {
			scaleTo((seq.getLength() * 2) + 1);
		}

	}

	/**
	 * Resizes the capacity to the given size
	 * 
	 * @param newSize
	 *            the size it will resize to
	 */
	private void scaleTo(int newSize) {
		capacity = newSize;
	}

}