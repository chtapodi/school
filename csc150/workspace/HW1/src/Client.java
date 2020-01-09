/**
 * The main method you use to run your experiments is here. You should study
 * this code to see what's going on and modify it in MANY WAYS to thoroughly
 * test each sort routine. See the assignment for suggestions of ways to modify
 * this code.
 * 
 * @author Chris Fernandes
 * @version 4/23/12
 */
public class Client {
	public static void main(String[] args) {
		int[] A, B = null;
		int n;
		long startTime;

		String timeString1="";
		String countString1="";
		
		String timeString2="";
		String countString2="";
		
		String timeString3="";
		String countString3="";
		
		String timeString4="";
		String countString4="";
		
		String timeString5="";
		String countString5="";
		
		String timeString6="";
		String countString6="";
		

		for (n = 100; n < 100000; n = n + 100) {
			
			// create a random array of n integers
			//System.out.println("\n");
			A = new int[n];
			for (int i = 0; i < n; i++)
				A[i] = (int) (-1000*Math.random());
			
			
			
			


			// time method5
		//	startTime = System.currentTimeMillis();
		//	B = Sort.method5(A);
			
			//timeString5+=System.currentTimeMillis() - startTime + "\n";
			//countString5+=(Sort.getMethod5Count() + "\n");
			
			if (!isSorted(B))
				System.out.println("  The array is not sorted.");
			
			
			System.out.println(n);


		}

		//System.out.println(""+countString5);
		
		

	}

	// -----------------------------
	/**
	 * prints the elements of array A to System.out. It prints 10 values per
	 * line
	 * 
	 * @param A
	 *            array to print
	 */
	private static void printArray(int[] A) {
		System.out.println();
		for (int i = 0; i < A.length; i++) {
			System.out.print(A[i] + ",  ");
			if ((i + 1) % 10 == 0)
				System.out.println();
		}
	}

	// -----------------------------
	/**
	 * tests array A to see whether it is sorted. Returns true if A is sorted
	 * from smallest to largest. If A is not sorted, it prints out the two
	 * unsorted elements that it found and then returns false.
	 * 
	 * @param A
	 *            the array to test
	 * @return true if sorted, false if not
	 */
	private static boolean isSorted(int[] A) {
		if (A == null)
			return true;
		for (int i = 0; i < A.length - 1; i++)
			if (A[i] > A[i + 1]) {
				System.out.print("A[" + i + "] = " + A[i] + " and A[" + (i + 1)
						+ "] = " + A[i + 1]);
				return false;
			}
		return true;
	}
}
