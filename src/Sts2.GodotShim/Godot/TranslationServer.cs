namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>TranslationServer</c>. The game only sets/gets
/// the active locale string; we hold it in memory since no translations are loaded.
/// </summary>
public static class TranslationServer
{
    private static string _locale = "en";

    public static void SetLocale(string locale) => _locale = locale;

    public static string GetLocale() => _locale;
}
