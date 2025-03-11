using System;
using System.Collections;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class PythonControlledMovement : MonoBehaviour
{
    TcpClient client;
    NetworkStream stream;
    byte[] buffer = new byte[1024];
    
    public float moveSpeed = 2f;
    public float turnAngle = 20f; // Fixed turn
    
    enum MoveType { None, Forward, TurnRight }
    MoveType moveCommand = MoveType.None;
    bool isTurning = false;

    void Start()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5005);
            stream = client.GetStream();
            stream.ReadTimeout = 50;
            ReceiveDataAsync(); // Start async receiving
        }
        catch (Exception e)
        {
            Debug.LogError("Connection failed: " + e.Message);
        }
    }

    void Update()
    {
        HandleMovement();
    }

    async void ReceiveDataAsync()
    {
        while (client.Connected)
        {
            if (stream.DataAvailable)
            {
                int bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
                string received = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                received = received.Length > 0 ? received[^1].ToString() : "";
                
                if (received == "1")
                    moveCommand = MoveType.Forward;
                else if (received == "0")
                    moveCommand = MoveType.TurnRight;
            }
            await Task.Delay(5); // Optimize delay
        }
    }

    void HandleMovement()
    {
        if (moveCommand == MoveType.Forward)
        {
            MoveForward();
            moveCommand = MoveType.None;
        }
        else if (moveCommand == MoveType.TurnRight && !isTurning)
        {
            isTurning = true;
            StartCoroutine(TurnSmoothly());
        }
    }

    void MoveForward()
    {
        transform.position += Vector3.forward * moveSpeed * Time.deltaTime;
    }

    IEnumerator TurnSmoothly()
    {
        float targetAngle = transform.eulerAngles.y + turnAngle;
        float startAngle = transform.eulerAngles.y;
        float elapsedTime = 0f;
        float duration = 0.2f;

        while (elapsedTime < duration)
        {
            float newAngle = Mathf.Lerp(startAngle, targetAngle, elapsedTime / duration);
            transform.rotation = Quaternion.Euler(0, newAngle, 0);
            elapsedTime += Time.deltaTime;
            yield return null;
        }

        transform.rotation = Quaternion.Euler(0, targetAngle, 0);
        isTurning = false;
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}
