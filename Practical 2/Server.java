import java.rmi.*;
import java.rmi.registry.*;

public class Server {
    public static void main(String[] args) {
        try {
            ConcatImpl obj = new ConcatImpl();
            LocateRegistry.createRegistry(2000); // Start RMI registry
            Naming.rebind("ConcatService", obj); // Bind the object
            System.out.println("Server is ready.");
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
