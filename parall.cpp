#include <iostream> 
#include <vector> 
#include <thread> 
#include <algorithm> 
 
using namespace std; 
int partition(vector<int>& arr, int low, int high) { 
    int pivot = arr[high]; 
    int i = low - 1; 
 
    for (int j = low; j < high; ++j) { 
        if (arr[j] < pivot) { 
            ++i; 
            swap(arr[i], arr[j]); 
        } 
    } 
    swap(arr[i + 1], arr[high]); 
    return i + 1; 
} 
 
// Parallel Quick Sort function 
void parallel_quick_sort(vector<int>& arr, int low, int high, int depth) { 
    if (low < high) { 
        int pivotIndex = partition(arr, low, high); 
 
        // If we still have parallelism depth left, use threads 
        if (depth > 0) { 
            thread leftThread(parallel_quick_sort, ref(arr), low, pivotIndex - 1, depth - 1); 
            thread rightThread(parallel_quick_sort, ref(arr), pivotIndex + 1, high, depth - 1); 
 
            leftThread.join(); 
            rightThread.join(); 
        } else { 
            parallel_quick_sort(arr, low, pivotIndex - 1, depth); 
            parallel_quick_sort(arr, pivotIndex + 1, high, depth); 
        } 
    } 
} 
 
 
 
 
 
Parallel Quick Sort 
 
int main() { 
    int n; 
    cout << "Enter the number of elements: "; 
    cin >> n; 
    vector<int> arr(n); 
    cout << "Enter the elements: "; 
    for (int i = 0; i < n; ++i) { 
        cin >> arr[i]; 
    } 
 
    int maxDepth = thread::hardware_concurrency(); // Set the maximum depth of parallelism 
    parallel_quick_sort(arr, 0, n - 1, maxDepth); 
 
    cout << "Sorted array: "; 
    for (int i = 0; i < n; ++i) { 
        cout << arr[i] << " "; 
    } 
    cout << endl; 
 
    return 0; 
} 
