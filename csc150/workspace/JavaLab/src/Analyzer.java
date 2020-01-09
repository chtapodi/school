/* Class to analyze simulated purchase trends from a department store.
 * Xavier Quinn
 * Lab Number One
 * 
 */


public class Analyzer {
	int empNum = 14;
	double money = 3400.59;
	char satis = 'B';
	boolean manStat = false;
	int i =0;
	int goodPurchase;
	
	
	public void storeInformation() {
	
		System.out.println("Current Number of Employees: " + empNum);
		System.out.println("Current Number amount of money: " + money);
		System.out.println("Customer Satisfaction level: " + satis);
		System.out.println("Is the manager in? " + manStat);
		
	}
	
	public void purchaseAnalyzer(double[] purchase, double min, double max) {
		
		if (purchase[0] > max && purchase[1] > max && purchase[2] > max) {
			System.out.println("Great Morning!");
		}
		else if (purchase[0] <= min && purchase[1] <= min && purchase[2] <= min) {
			System.out.println("Awful Morning!");
		}
		
	}
	
	public void inDepthAnalyzer(double[] purchase, double min, double max) {
		
		while (purchase[i] <= max && i < 9) {
			i++;
			
		}
		if (i>9) {
			System.out.println("There have been no good purchases");
		}
		else {
			System.out.println("The first good purchase was: " + purchase[i]);
		}
		
		i=0;
		
		while (purchase[i] > min && i < 9) {
			i++;
			if (purchase[i] > max) {
				goodPurchase++;
			}
		}
		if(i<9) {
			System.out.println("There have been " + goodPurchase + " good sales before the first bad one");
		}
		else {
			System.out.println("There have been " + goodPurchase + " good  sales and no bad ones");
		}
	}
	
	/** a really dumb way of printing the entire array
	 * 
	 *  purchase is a chronological collection of purchases
	 */
	public void printer(double[] purchase) {
		System.out.println();
		System.out.println("purchase[0]: " + purchase[0]);
		System.out.println("purchase[1]: " + purchase[1]);
		System.out.println("purchase[2]: " + purchase[2]);
		System.out.println("purchase[3]: " + purchase[3]);
		System.out.println("purchase[4]: " + purchase[4]);
		System.out.println("purchase[5]: " + purchase[5]);
		System.out.println("purchase[6]: " + purchase[6]);
		System.out.println("purchase[7]: " + purchase[7]);
		System.out.println("purchase[8]: " + purchase[8]);
		System.out.println("purchase[9]: " + purchase[9]);
	}
	
}
