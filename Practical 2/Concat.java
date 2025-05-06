import java.rmi.*;

public interface Concat extends Remote {
    String join(String s1, String s2) throws RemoteException;
}
