

dataset_df = pd.read_csv("../train_dataset.csv")

features = dataset_df.drop(columns=['slice Type']).to_numpy()
label = dataset_df['slice Type'].to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

features_tensor = torch.tensor(features, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long) - 1

X_train, X_test, y_train, y_test = train_test_split(features_tensor, label_tensor, test_size=0.2, random_state=42, stratify=label)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size=512

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_size=16, hidden_units=32, dropout=0.4, num_classes=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.05)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    correct_prediction = 0
    total_checked = 0

    for batch_X, batch_y in train_loader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_prediction += (predicted == batch_y).sum().item()
        total_checked += batch_y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_prediction / total_checked   
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        

model.eval()
correct_prediction = 0
total_checked = 0

with torch.no_grad():

    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        correct_prediction += (predicted == batch_y).sum().item()
        total_checked += batch_y.size(0)


accuracy = 100 * correct_prediction / total_checked

print(f'accuracy {accuracy}%')
